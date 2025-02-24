import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
from torchvision.io import read_image
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from dataloader import Comp0249Dataset, Comp0249DatasetYolo
from tqdm import tqdm

from utils import check_binary

# Build Dert manually 

class TotalModel(nn.Module):
    def __init__(self, S=7, B=2, C=20, w=960, h=720, coeffcient_lambda_coord=5, coeffcient_lambda_noobj=0.5):
        """
        in_channels: 输入图像的通道数
        num_classes: 实际类别数量，不含背景，默认5
        """
        super(TotalModel, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.w = w
        self.h = h
        self.coeffcient_lambda_coord = coeffcient_lambda_coord
        self.coeffcient_lambda_noobj = coeffcient_lambda_noobj

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(192, 128, kernel_size=1), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            self._make_conv_block(512, 1024, num_repeats=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._make_conv_block(1024, 1024, num_repeats=2),
            nn.AdaptiveAvgPool2d((7, 7))  # 强制将输出调整为 (batch, 1024, 3, 3)

        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, 7 * 7 * (C + B * 5))  # 输出 final_size x final_size x (类别数+边界框数*5)
        )

    def _make_conv_block(self, in_channels, out_channels, num_repeats):
        layers = []
        for _ in range(num_repeats):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(0.1))
            in_channels = out_channels   # 更新输入通道为当前层的输出通道
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, 3, w, h)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, 7, 7, self.C + self.B * 5)
        return x


def yolo_loss_func(predictions, targets, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
    """
    YOLO 损失函数计算（支持 batch 维度）
    
    参数：
        predictions: 模型预测值, 形状 (N, S, S, C + B*5)
        targets: 真实标签, 形状 (N, S, S, C + B*5)
        S: 网格划分数
        B: 每个网格预测的边界框数量
        C: 类别数
        lambda_coord: 坐标损失系数
        lambda_noobj: 无目标置信度损失系数

    返回：
        total_loss: 总损失
    """
    assert predictions.shape == targets.shape, "预测值与目标值形状不一致"
    assert predictions.shape[-1] == C + B * 5, "预测值与目标值通道数不一致"

    batch = predictions.shape[0]  # 获取 batch 大小

    # 提取类别概率
    pred_class_probs = predictions[..., :C]  # (N, S, S, C)
    target_class_probs = targets[..., :C]      # (N, S, S, C)

    # 提取目标掩码
    # 此处假设每个 box 信息占 5 个通道，目标存在则置信度 > 0
    obj_mask = (targets[..., C+4::5].max(dim=-1, keepdim=True)[0] > 0).float()  # (N, S, S, 1)
    noobj_mask = 1 - obj_mask  # (N, S, S, 1)

    # 将真实边界框 reshape 为 (N, S, S, B, 5)
    target_boxes = targets[..., C:].view(batch, S, S, B, 5)
    # 同理，将预测边界框 reshape 为 (N, S, S, B, 5)
    pred_boxes = predictions[..., C:].view(batch, S, S, B, 5)

    pred_xy = pred_boxes[..., :2]   # (N, S, S, B, 2)
    pred_wh = pred_boxes[..., 2:4]   # (N, S, S, B, 2)
    pred_conf = pred_boxes[..., 4:5] # (N, S, S, B, 1)

    target_xy = target_boxes[..., :2]   # (N, S, S, B, 2)
    target_wh = target_boxes[..., 2:4]   # (N, S, S, B, 2)
    target_conf = target_boxes[..., 4:5] # (N, S, S, B, 1)

    # 计算 IoU 以选择最佳匹配的边界框
    iou_scores = bbox_iou(target_boxes[..., :4], pred_boxes[..., :4])  # 形状 (N, S, S, B)
    best_iou, best_bbox = iou_scores.max(dim=-1, keepdim=True)  # 形状 (N, S, S, 1)
    # 例如 best_bbox[i][j][k][0] = 1 表示第 2 个边界框是最佳匹配
    # best_bbox[i][j][k][0] = 0 表示第 1 个边界框是最佳匹配

    assert check_binary(best_bbox[..., 0]), "best_bbox 中不全为 0 或 1"
    assert best_bbox.shape == (batch, S, S, 1)

    # 扩展目标掩码到边界框维度，此处先对 best_bbox 进行 one-hot 编码
    best_bbox_mask = F.one_hot(best_bbox.squeeze(-1), B).unsqueeze(-1).float()  # (N, S, S, B, 1)
    assert best_bbox_mask.shape == (batch, S, S, B, 1)

    # 根据 best_bbox_mask 从 B 个 box 中提取最佳的框信息（对 B 维求和）
    pred_best_xy   = (best_bbox_mask * pred_xy).sum(dim=-2)  # (N, S, S, 2)
    pred_best_wh   = (best_bbox_mask * pred_wh).sum(dim=-2)  # (N, S, S, 2)
    pred_best_conf = (best_bbox_mask * pred_conf).sum(dim=-2)  # (N, S, S, 1)

    target_best_xy   = (best_bbox_mask * target_xy).sum(dim=-2)  # (N, S, S, 2)
    target_best_wh   = (best_bbox_mask * target_wh).sum(dim=-2)  # (N, S, S, 2)
    target_best_conf = (best_bbox_mask * target_conf).sum(dim=-2)  # (N, S, S, 1)

    # 计算坐标损失（仅计算目标框的损失）
    xy_loss = F.mse_loss(obj_mask * pred_best_xy, obj_mask * target_best_xy, reduction='sum')
    wh_loss = F.mse_loss(
        obj_mask * torch.sqrt(torch.clamp(pred_best_wh, min=0) + 1e-6),
        obj_mask * torch.sqrt(torch.clamp(target_best_wh, min=0) + 1e-6),
        reduction='sum'
    )

    # 计算置信度损失（分目标与无目标部分）
    obj_conf_loss = F.mse_loss(obj_mask * pred_best_conf, obj_mask * target_best_conf, reduction='sum')
    noobj_conf_loss = F.mse_loss(noobj_mask * pred_best_conf, noobj_mask * target_best_conf, reduction='sum')

    # 计算分类损失（仅对目标存在的网格）
    class_loss = F.mse_loss(obj_mask * pred_class_probs, obj_mask * target_class_probs, reduction='sum')

    # 最终总损失
    total_loss = (
        lambda_coord * xy_loss +       # 坐标损失
        lambda_coord * wh_loss +       # 宽高损失
        obj_conf_loss +                # 目标置信度损失
        lambda_noobj * noobj_conf_loss +# 无目标置信度损失
        class_loss                     # 分类损失
    )

    return total_loss

def bbox_iou(box1, box2):
    """
    计算两个边界框的 IoU（交并比）
    
    box1, box2: 形状均为 (S, S, B, 4)，格式为 [x, y, w, h]
    返回: IoU, 形状为 (S, S, B)
    """
    box1_x1 = box1[..., 0] - box1[..., 2] / 2
    box1_y1 = box1[..., 1] - box1[..., 3] / 2
    box1_x2 = box1[..., 0] + box1[..., 2] / 2
    box1_y2 = box1[..., 1] + box1[..., 3] / 2

    box2_x1 = box2[..., 0] - box2[..., 2] / 2
    box2_y1 = box2[..., 1] - box2[..., 3] / 2
    box2_x2 = box2[..., 0] + box2[..., 2] / 2
    box2_y2 = box2[..., 1] + box2[..., 3] / 2

    # 计算交集
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # 计算并集
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    union_area = box1_area + box2_area - inter_area + 1e-6

    return inter_area / union_area


def yolo_accuracy(predictions, targets, C=20):
    """
    计算 YOLO 模型在存在目标格子上的分类准确率
    
    参数：
        predictions: 张量，形状为 (S, S, C + 2B)，模型预测输出
        targets: 张量，形状为 (S, S, C + 2B)，真实标签
        C: 类别数
    
    返回：
        accuracy: 针对存在目标的格子的分类准确率，介于 0~1 之间
    """
    # 掩码：判断哪些网格单元存在目标（targets[..., C] 代表目标置信度）
    obj_mask = targets[..., C] > 0  # 形状 (S, S)
    
    # 如果没有目标，则返回 0.0 的准确率
    if obj_mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    # 预测类别：取前 C 维，并在类别维度上取 argmax 得到预测类别
    pred_class = torch.argmax(predictions[..., :C], dim=-1)  # (S, S)
    
    # 真实类别：从 targets 中取前 C 维，并取 argmax 得到真实类别
    true_class = torch.argmax(targets[..., :C], dim=-1)      # (S, S)
    
    # 仅计算存在目标的网格的分类正确率
    correct = (pred_class[obj_mask] == true_class[obj_mask]).float().sum()
    total = obj_mask.sum().float()
    
    accuracy = correct / total
    return accuracy

# 使用示例
# if __name__ == "__main__":
#     preds = torch.randn(2, 7, 7, 30)  # batch_size=2, 网格 7x7, 30 维输出 (例如 C=20，B=2, 20+2*5=30)
#     targets = torch.randn(2, 7, 7, 30)
#     loss_value = yolo_loss_func(preds, targets)
#     print("Loss:", loss_value.item())





if __name__ == "__main__":


    train_dataset = Comp0249DatasetYolo('data/CamVid', "train", scale=1, transform=None, target_transform=None)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    data = train_dataset[0]
    # print(data[0].shape, data[1].shape)    
    image_size = data[0].shape # (3, h, w)     
    label_size = data[1].unsqueeze(0).shape # (7, 7, C + 2B)
    tst_label = data[1].unsqueeze(0)
    model = TotalModel(S=7, B=2, C=5, w=image_size[2], h=image_size[1])


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_loss = []
    total_acc = []
    num_epochs = 10

    # patience = 3         # 容忍连续多少个 epoch 损失不降
    # min_delta = 0.001    # 损失需要下降的最小改变量
    # best_loss = float('inf')
    # epochs_no_improve = 0

    test = yolo_loss_func(tst_label, tst_label, S=model.S, B=model.B, C=model.C)
    print(test)
    test_acc = yolo_accuracy(tst_label, tst_label, C=model.C)
    print(test_acc)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        loss_per_epoch = 0.0
        acc_per_epoch = 0.0

        for images, labels in tqdm(train_loader, desc="Batches"):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float)

            optimizer.zero_grad()

            # 计算预测结果
            pred = model(images)

            loss = yolo_loss_func(pred, labels, S=model.S, B=model.B, C=model.C)
            loss.backward()
            optimizer.step()

            loss_per_epoch += loss.item()

            # 使用刚刚定义的 yolo_accuracy 计算存在目标格子的分类准确率
            batch_acc = yolo_accuracy(pred, labels, C=model.C)
            acc_per_epoch += batch_acc.item()

        loss_per_epoch /= len(train_loader)
        acc_per_epoch /= len(train_loader)
        print(f"Epoch: {epoch}, Loss: {loss_per_epoch:.4f}, Acc: {acc_per_epoch:.4f}")

        total_loss.append(loss_per_epoch)
        total_acc.append(acc_per_epoch)

        # Early stopping
        # if best_loss - loss_per_epoch > min_delta:
        #     best_loss = loss_per_epoch
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1

        # if epochs_no_improve >= patience:
        #     print("Early stopping triggered.")
        #     break



    # 保存模型及训练指标
    torch.save(model.state_dict(), 'results/model.pth')
    torch.save(model, 'results/full_model.pth')
    import json
    data = {
        'loss': total_loss,
        'accuracy': total_acc
    }
    with open('results/data.json', 'w') as f:
        json.dump(data, f)
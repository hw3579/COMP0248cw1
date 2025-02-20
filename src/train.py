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


# 定义计算 IoU 的内部函数，输入框格式为 (x, y, w, h)
def iou(box1, box2):
        # 计算左上角和右下角
        box1_x1 = box1[..., 0] - box1[..., 2] / 2
        box1_y1 = box1[..., 1] - box1[..., 3] / 2
        box1_x2 = box1[..., 0] + box1[..., 2] / 2
        box1_y2 = box1[..., 1] + box1[..., 3] / 2
        
        box2_x1 = box2[..., 0] - box2[..., 2] / 2
        box2_y1 = box2[..., 1] - box2[..., 3] / 2
        box2_x2 = box2[..., 0] + box2[..., 2] / 2
        box2_y2 = box2[..., 1] + box2[..., 3] / 2
        
        inter_x1 = torch.max(box1_x1, box2_x1)
        inter_y1 = torch.max(box1_y1, box2_y1)
        inter_x2 = torch.min(box1_x2, box2_x2)
        inter_y2 = torch.min(box1_y2, box2_y2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = area1 + area2 - inter_area + 1e-6  # 避免除零
        return inter_area / union_area

def yolo_loss_func(predictions, targets, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
    """
    predictions: (batch_size, S, S, C + B*5) 预测值
    targets: (batch_size, S, S, C + B*5) 真实标签
    """
    batch_size = predictions.shape[0]
    # 定义 MSE 损失，reduction 为 "sum"
    mse_loss = lambda pred, target: F.mse_loss(pred, target, reduction="sum")
    
    # predictions = torch.cat([predictions[..., C:], predictions[..., :C]], dim=-1)
    # targets = torch.cat([targets[..., C:], targets[..., :C]], dim=-1)

    # 掩码：判断每个格子是否存在目标（使用第一个边界框的置信度判断）
    obj_mask = targets[..., C+4] > 0   # 目标存在
    no_obj_mask = ~obj_mask          # 目标不存在
    # 计算预测的两个边界框与真实框（targets[...,5:9]）的 IoU
    iou_b1 = iou(predictions[..., 5:9], targets[..., 5:9])
    iou_b2 = iou(predictions[..., 10:14], targets[..., 5:9])
    ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)  # (2, batch_size, S, S)
    best_box = torch.argmax(ious, dim=0)  # (batch_size, S, S)，值为 0 或 1
    box_mask = (best_box == 0).float().unsqueeze(-1)  # (batch_size, S, S, 1)

    # 根据最佳 IoU 选择对应的边界框预测
    box_predictions = box_mask * predictions[..., 5:9] + (1 - box_mask) * predictions[..., 10:14]
    box_targets = targets[..., 5:9]

    # 对宽和高取 sqrt，保证数值稳定性
    box_predictions = box_predictions.clone()
    box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
    box_targets = box_targets.clone()
    box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

    # 1. 定位损失
    localization_loss = mse_loss(box_predictions[obj_mask], box_targets[obj_mask])

    # 2. 置信度损失：选择负责预测的置信度
    conf_pred = box_mask * predictions[..., 4:5] + (1 - box_mask) * predictions[..., 9:10]
    conf_loss_obj = mse_loss(conf_pred[obj_mask], targets[..., 4:5][obj_mask])
    conf_loss_noobj = mse_loss(predictions[..., 4:5][no_obj_mask], targets[..., 4:5][no_obj_mask])
    conf_loss_noobj += mse_loss(predictions[..., 9:10][no_obj_mask], targets[..., 4:5][no_obj_mask])

    # 3. 分类损失：仅在存在物体的格子上计算
    class_loss = mse_loss(predictions[..., :C][obj_mask], targets[..., :C][obj_mask])
    
    # 综合损失
    total_loss = (lambda_coord * localization_loss
                  + conf_loss_obj
                  + lambda_noobj * conf_loss_noobj
                  + class_loss)
    
    return total_loss / batch_size  # 归一化到 batch 维度


def yolo_accuracy(predictions, targets, C=20):
    """
    计算 YOLO 模型在存在目标格子上的分类准确率
    参数：
        predictions: 张量，形状为 (batch_size, S, S, C + B*5)，模型预测输出
        targets: 张量，形状为 (batch_size, S, S, C + B*5)，真实标签
        C: 类别数
    返回：
        accuracy: 针对存在目标的格子的分类准确率，介于0~1之间
    """
    # 掩码：判断哪些格子存在目标（使用 targets[..., 4] > 0 判定）
    obj_mask = targets[..., 4] > 0  # 形状 (batch_size, S, S)
    
    # 如果没有目标，则返回 0 准确率
    if obj_mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    # 预测类别：取前 C 个数，argmax 得到预测类别
    pred_class = torch.argmax(predictions[..., :C], dim=-1)  # 形状 (batch_size, S, S)
    # 真实类别：同样从 targets 中取前 C 个数
    true_class = torch.argmax(targets[..., :C], dim=-1)       # 形状 (batch_size, S, S)
    
    # 仅对存在目标的格子计算准确率
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
    data_size = data[0].shape
    model = TotalModel(S=7, B=2, C=5, w=data_size[2], h=data_size[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_loss = []
    total_acc = []
    num_epochs = 15

    # patience = 3         # 容忍连续多少个 epoch 损失不降
    # min_delta = 0.001    # 损失需要下降的最小改变量
    # best_loss = float('inf')
    # epochs_no_improve = 0

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
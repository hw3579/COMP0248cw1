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


class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky(x)
        return x


class ResUnit(nn.Module):
    def __init__(self, in_channels):
        super(ResUnit, self).__init__()
        self.cbl1 = CBL(in_channels, in_channels // 2, 1, 1, 0)
        self.cbl2 = CBL(in_channels // 2, in_channels, 3, 1, 1)
    
    def forward(self, x):
        residual = x
        x = self.cbl1(x)
        x = self.cbl2(x)
        return x + residual
    
class ResUnitX(nn.Module):
    def __init__(self, in_channels, X):
        super(ResUnitX, self).__init__()
        self.conv1 = CBL(in_channels, in_channels*2, 3, 2, 1)
        self.resunit = ResUnit(in_channels*2)
        self.X = X

    def forward(self, x):
        x = self.conv1(x)
        for i in range(self.X):
            x = self.resunit(x)

        return x
    

class Yolov3(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(Yolov3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = num_classes + 5*2
        self.conv1 = CBL(in_channels, 32, 3, 1, 1)
        self.resunit1 = ResUnitX(32, 1)
        self.resunit2 = ResUnitX(64, 2)
        self.resunit3 = ResUnitX(128, 8)
        self.resunit4 = ResUnitX(256, 8)
        self.resunit5 = ResUnitX(512, 4)
        # 预定义所有需要的模块
        self.cblset1 = self._CBLset(1024)
        self.pred1_cbl = CBL(512, 1024, 3, 1, 1)
        self.pred1_conv = nn.Conv2d(1024, self.out_channels, 1, 1, 0)
        
        self.pred1_down = nn.Conv2d(512, 256, 1, 1, 0)
        self.cblset2 = self._CBLset(768)
        self.pred2_cbl = CBL(384, 768, 3, 1, 1)
        self.pred2_conv = nn.Conv2d(768, self.out_channels, 1, 1, 0)
        
        self.pred2_down = nn.Conv2d(384, 192, 1, 1, 0)
        self.cblset3 = self._CBLset(448)
        self.pred3_cbl = CBL(224, 448, 3, 1, 1)
        self.pred3_conv = nn.Conv2d(448, self.out_channels, 1, 1, 0)

    def _CBLset(self, in_channels):
        return nn.Sequential(
            CBL(in_channels, in_channels//2, 1, 1, 0),
            CBL(in_channels//2, in_channels, 3, 1, 1),
            CBL(in_channels, in_channels//2, 1, 1, 0),
            CBL(in_channels//2, in_channels, 3, 1, 1),
            CBL(in_channels, in_channels//2, 1, 1, 0)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.resunit1(x)
        x = self.resunit2(x)
        pred3 = self.resunit3(x)
        pred2 = self.resunit4(pred3)
        x = self.resunit5(pred2)
        pred1 = self.cblset1(x)
        
        #### pred1
        pred1_out = self.pred1_cbl(pred1)
        pred1_out = self.pred1_conv(pred1_out)

        #### pred2
        pred1 = self.pred1_down(pred1)
        pred1 = F.interpolate(pred1, pred2.size()[-2:], mode='nearest')
        pred2 = torch.cat((pred1, pred2), 1)
        pred2 = self.cblset2(pred2)

        pred2_out = self.pred2_cbl(pred2)
        pred2_out = self.pred2_conv(pred2_out)

        #### pred3
        pred2 = self.pred2_down(pred2)
        pred2 = F.interpolate(pred2, pred3.size()[-2:], mode='nearest')
        pred3 = torch.cat((pred2, pred3), 1)
        pred3 = self.cblset3(pred3)

        pred3_out = self.pred3_cbl(pred3)
        pred3_out = self.pred3_conv(pred3_out)

        pred1_out = pred1_out.permute(0, 2, 3, 1)
        pred2_out = pred2_out.permute(0, 2, 3, 1)
        pred3_out = pred3_out.permute(0, 2, 3, 1)
        
        return pred1_out, pred2_out, pred3_out
    

def test_CBL():
    cbl = CBL(3, 32, 3, 1, 1)
    x = torch.randn(1, 3, 416, 416)
    y = cbl(x)
    print(y.shape)

def test_ResUnit():
    resunit = ResUnit(32)
    x = torch.randn(1, 32, 416, 416)
    y = resunit(x)
    print(y.shape)

def test_ResUnitX():
    resunit = ResUnitX(32, 1)
    x = torch.randn(1, 32, 416, 416)
    y = resunit(x)
    print(y.shape)

def test_Yolov3():
    yolov3 = Yolov3(5)
    x = torch.randn(1, 3, 960, 720)
    y1, y2, y3 = yolov3(x)
    print(y1.shape, y2.shape, y3.shape)

from train import bbox_iou
def yolo_loss_funcv3_1(predictions, targets, Sx=7, Sy=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
    """
    YOLO 损失函数计算（支持 batch 维度）
    
    参数：
        predictions: 模型预测值, 形状 (N, Sy, Sx, C + B*5)
        targets: 真实标签, 形状 (N, Sy, Sx, C + B*5)
        Sx: 网格水平方向划分数
        Sy: 网格垂直方向划分数
        B: 每个网格预测的边界框数量
        C: 类别数
        lambda_coord: 坐标损失系数
        lambda_noobj: 无目标置信度损失系数

    返回：
        total_loss: 总损失
    """
    assert predictions.shape == targets.shape, "预测值与目标值形状不一致"
    assert predictions.shape[-1] == C + B * 5, "预测值与目标值通道数不一致"

    Sx = math.ceil(Sx)
    Sy = math.ceil(Sy)

    batch = predictions.shape[0]  # 获取 batch 大小

    # 提取类别概率
    pred_class_probs = predictions[..., :C]  # (N, Sy, Sx, C)
    target_class_probs = targets[..., :C]      # (N, Sy, Sx, C)

    # 提取目标掩码
    # 此处假设每个 box 信息占 5 个通道，目标存在则置信度 > 0
    obj_mask = (targets[..., C+4::5].max(dim=-1, keepdim=True)[0] > 0).float()  # (N, Sy, Sx, 1)
    noobj_mask = 1 - obj_mask  # (N, Sy, Sx, 1)

    # 将真实边界框 reshape 为 (N, Sy, Sx, B, 5)
    target_boxes = targets[..., C:].view(batch, Sy, Sx, B, 5)
    # 同理，将预测边界框 reshape 为 (N, Sy, Sx, B, 5)
    pred_boxes = predictions[..., C:].view(batch, Sy, Sx, B, 5)

    pred_xy = pred_boxes[..., :2]   # (N, Sy, Sx, B, 2)
    pred_wh = pred_boxes[..., 2:4]   # (N, Sy, Sx, B, 2)
    pred_conf = pred_boxes[..., 4:5] # (N, Sy, Sx, B, 1)

    target_xy = target_boxes[..., :2]   # (N, Sy, Sx, B, 2)
    target_wh = target_boxes[..., 2:4]   # (N, Sy, Sx, B, 2)
    target_conf = target_boxes[..., 4:5] # (N, Sy, Sx, B, 1)

    # 计算 IoU 以选择最佳匹配的边界框
    iou_scores = bbox_iou(target_boxes[..., :4], pred_boxes[..., :4])  # 形状 (N, Sy, Sx, B)
    best_iou, best_bbox = iou_scores.max(dim=-1, keepdim=True)  # 形状 (N, Sy, Sx, 1)
    # 例如 best_bbox[i][j][k][0] = 1 表示第 2 个边界框是最佳匹配
    # best_bbox[i][j][k][0] = 0 表示第 1 个边界框是最佳匹配

    assert best_bbox.shape == (batch, Sy, Sx, 1)

    # 扩展目标掩码到边界框维度，此处先对 best_bbox 进行 one-hot 编码
    best_bbox_mask = F.one_hot(best_bbox.squeeze(-1), B).unsqueeze(-1).float()  # (N, Sy, Sx, B, 1)
    assert best_bbox_mask.shape == (batch, Sy, Sx, B, 1)

    # 根据 best_bbox_mask 从 B 个 box 中提取最佳的框信息（对 B 维求和）
    pred_best_xy   = (best_bbox_mask * pred_xy).sum(dim=-2)  # (N, Sy, Sx, 2)
    pred_best_wh   = (best_bbox_mask * pred_wh).sum(dim=-2)  # (N, Sy, Sx, 2)
    pred_best_conf = (best_bbox_mask * pred_conf).sum(dim=-2)  # (N, Sy, Sx, 1)

    target_best_xy   = (best_bbox_mask * target_xy).sum(dim=-2)  # (N, Sy, Sx, 2)
    target_best_wh   = (best_bbox_mask * target_wh).sum(dim=-2)  # (N, Sy, Sx, 2)
    target_best_conf = (best_bbox_mask * target_conf).sum(dim=-2)  # (N, Sy, Sx, 1)

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

def yolo_loss_funcv3(pred, target, Sx, Sy, B=2, C=20):
    # 假设 pred 和 target 是三个尺度的预测和目标
    large_cell = yolo_loss_funcv3_1(pred[0], target[0], Sx=Sx, Sy=Sy, B=B, C=C)
    medium_cell = yolo_loss_funcv3_1(pred[1], target[1], Sx=Sx*2, Sy=Sy*2, B=B, C=C)
    small_cell = yolo_loss_funcv3_1(pred[2], target[2], Sx=Sx*4, Sy=Sy*4, B=B, C=C)
    
    return (large_cell + medium_cell + small_cell)/3



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

def yolo_accuracy_v3(pred, target, C=20):
    # 假设 pred 和 target 是三个尺度的预测和目标
    large_cell = yolo_accuracy(pred[0], target[0], C)
    medium_cell = yolo_accuracy(pred[1], target[1], C)
    small_cell = yolo_accuracy(pred[2], target[2], C)
    
    return (large_cell + medium_cell + small_cell) / 3
   

import math
if __name__ == '__main__':
    # test_ResUnit() # 1, 32, 416, 416
    # test_ResUnitX() # 1, 64, 208, 208
    # test_Yolov3() # 1, 5, 13, 13

    # 960, 720 -> 30, 23 & 60, 45 & 120, 90


    train_dataset = Comp0249DatasetYolo('data/CamVid', "train", scale=1, transform=None, target_transform=None, version=3)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    data = train_dataset[0]
    # print(data[0].shape, data[1].shape)    
    image_size = data[0].shape # (3, h, w)  
    _, h, w = image_size   

    tst_label = [None, None, None]
    tst_label[0] = data[1][0].unsqueeze(0)
    tst_label[1] = data[1][1].unsqueeze(0)
    tst_label[2] = data[1][2].unsqueeze(0)
    model = Yolov3(5, in_channels=3)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_loss = []
    total_acc = []
    num_epochs = 400

    # patience = 3         # 容忍连续多少个 epoch 损失不降
    # min_delta = 0.001    # 损失需要下降的最小改变量
    # best_loss = float('inf')
    # epochs_no_improve = 0

    test = yolo_loss_funcv3(tst_label, tst_label, w/32, h/32, C=5)
    print(test)
    test_acc = yolo_accuracy_v3(tst_label, tst_label, C=5)
    print(test_acc)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        loss_per_epoch = 0.0
        acc_per_epoch = 0.0

        for images, labels in tqdm(train_loader, desc="Batches"):
            images = images.to(device, dtype=torch.float32)
            for i in range(len(labels)):
                labels[i] = labels[i].to(device, dtype=torch.float)

            optimizer.zero_grad()

            # 计算预测结果
            pred = model(images)

            loss = yolo_loss_funcv3(pred, labels, w/32, h/32, C=5)
            loss.backward()
            optimizer.step()

            loss_per_epoch += loss.item()

            # 使用刚刚定义的 yolo_accuracy 计算存在目标格子的分类准确率
            batch_acc = yolo_accuracy_v3(pred, labels, C=5)
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

        if loss_per_epoch < 0.1:
            break


    # 保存模型及训练指标
    torch.save(model, 'results/full_model_yolov3.pth')
    import json
    data = {
        'loss': total_loss,
        'accuracy': total_acc
    }
    with open('results/data_yolov3.json', 'w') as f:
        json.dump(data, f)
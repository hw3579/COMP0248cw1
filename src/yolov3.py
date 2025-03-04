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

from dataloader import Comp0249Dataset
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
        # self.dropout = nn.Dropout2d(0.3)   # 添加空间Dropout

    def forward(self, x):
        residual = x
        x = self.cbl1(x)
        x = self.cbl2(x)
        # x = self.dropout(x)  # 在residual连接前应用dropout
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
    def __init__(self, num_classes, in_channels=3, B=2):
        super(Yolov3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = num_classes + 5*B
        self.B = B
        self.C = num_classes
        # self.conv1 = CBL(in_channels, 32, 3, 1, 1)
        self.conv1 = CBL(in_channels, 128, 3, 1, 1)

        # 1 2 8 8 4 -> 0 0 2 2 1 -> 0 0 1 1 1
        # self.resunit1 = ResUnitX(32, 1)
        # self.resunit2 = ResUnitX(64, 2)
        self.resunit3 = ResUnitX(128, 1)
        self.resunit4 = ResUnitX(256, 1)
        self.resunit5 = ResUnitX(512, 1)


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

        self._initialize_weights()


    def _initialize_weights(self):
        """初始化网络权重，确保输出在合理范围内"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 对大多数卷积层使用He/Kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        
        # 特别处理预测层的卷积（这些层直接输出边界框参数和置信度）
        for pred_layer in [self.pred1_conv, self.pred2_conv, self.pred3_conv]:
            # 对于输出层，使用较小方差的初始化，使初始输出接近零
            nn.init.normal_(pred_layer.weight, mean=0.0, std=0.01)
            if pred_layer.bias is not None:
                # 为类别和置信度通道设置特殊初始值
                # 对于sigmoid激活函数，设置为0表示初始输出为0.5
                nn.init.constant_(pred_layer.bias, 0.0)
                
                # 可选：为置信度通道设置负偏置，使初始置信度较低
                # 以下代码设置置信度通道的初始偏置为-2，使sigmoid输出约为0.12
                for i in range(self.B):
                    # 假设每个边界框有5个值(x,y,w,h,conf)，类别在前面
                    conf_index = self.C + i * 5 + 4
                    if conf_index < len(pred_layer.bias):
                        # 设置所有置信度通道的偏置
                        pred_layer.bias.data[conf_index::self.out_channels] = -2.0
 

    def _CBLset(self, in_channels):
        return nn.Sequential(
            CBL(in_channels, in_channels//2, 1, 1, 0),
            # CBL(in_channels//2, in_channels, 3, 1, 1),
            # nn.Dropout2d(0.1),  # 添加dropout
            # CBL(in_channels, in_channels//2, 1, 1, 0),
            # CBL(in_channels//2, in_channels, 3, 1, 1),
            # nn.Dropout2d(0.1),  # 添加dropout
            # CBL(in_channels, in_channels//2, 1, 1, 0)
        )

    def forward(self, x):
        x = self.conv1(x)
        # x = self.resunit1(x)
        # x = self.resunit2(x)

        x = F.interpolate(x, scale_factor=0.25, mode='nearest') #大小匹配
        pred3 = self.resunit3(x)
        pred2 = self.resunit4(pred3)
        x = self.resunit5(pred2)
        pred1 = self.cblset1(x)
        
        # 添加dropout
        # pred1 = F.dropout(pred1, p=0.2, training=self.training)
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
    yolov3 = Yolov3(5, in_channels=3, B=1)
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
    # 处理可能的NaN或Inf值
    iou_scores = torch.nan_to_num(iou_scores, nan=0.0, posinf=1.0, neginf=0.0)

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



    # 激活函数
    # pred_best_xy = torch.sigmoid(pred_best_xy)  # 限制xy在0-1范围内
    # pred_best_wh = torch.exp(pred_best_wh)      # 确保宽高为正值
    pred_best_conf = torch.sigmoid(pred_best_conf)
    target_best_conf = torch.sigmoid(target_best_conf)
    pred_class_probs = torch.sigmoid(pred_class_probs)  # 类别概率使用 sigmoid 激活函数


    # 计算坐标损失（仅计算目标框的损失）
    xy_loss = F.mse_loss(obj_mask * pred_best_xy, obj_mask * target_best_xy, reduction='sum')
    wh_loss = F.mse_loss(
        obj_mask * torch.sqrt(torch.clamp(pred_best_wh, min=0) + 1e-6),
        obj_mask * torch.sqrt(torch.clamp(target_best_wh, min=0) + 1e-6),
        reduction='sum'
    )

    # 计算置信度损失（分目标与无目标部分）
    obj_conf_loss = F.binary_cross_entropy(obj_mask * pred_best_conf, obj_mask * target_best_conf, reduction='sum')
    noobj_conf_loss = F.binary_cross_entropy(noobj_mask * pred_best_conf, noobj_mask * target_best_conf, reduction='sum')

    # 计算分类损失（仅对目标存在的网格）
    class_loss = F.binary_cross_entropy(obj_mask * pred_class_probs, obj_mask * target_class_probs, reduction='sum')

    print(xy_loss.item(), wh_loss.item(), obj_conf_loss.item(), noobj_conf_loss.item(), class_loss.item())
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
    
    return 0.4 * large_cell + 0.4 * medium_cell + 0.2 * small_cell



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
    
    # 计算位置准确度 - 使用相对误差的倒数，并处理除零情况
    position_pred = predictions[..., C:C+2]  # 只取xy坐标
    position_target = targets[..., C:C+2]    # 真实xy坐标
    # 添加小值防止除零
    epsilon = 1e-6
    # 计算相对误差并转换为准确度值
    xy_error = torch.abs(position_pred - position_target) / (torch.abs(position_target) + epsilon)
    xy_error = xy_error[obj_mask].mean()  # 使用mean代替sum/total更直观
    # 将误差转换为0-1范围的准确度值
    xy_accuracy = torch.clamp(1.0 - xy_error, 0.0, 1.0)

    # 同样处理宽高误差
    wh_pred = predictions[..., C+2:C+4]
    wh_target = targets[..., C+2:C+4]
    wh_error = torch.abs(wh_pred - wh_target) / (torch.abs(wh_target) + epsilon)
    wh_accuracy = torch.clamp(1.0 - wh_error[obj_mask].mean(), 0.0, 1.0)

    # 置信度准确度
    conf_error = torch.abs(predictions[..., C+4] - targets[..., C+4]) / (torch.abs(targets[..., C+4]) + epsilon)
    conf_accuracy = torch.clamp(1.0 - conf_error[obj_mask].mean(), 0.0, 1.0)

    # 返回综合准确度（可以加权平均）
    return torch.tensor([accuracy.item(), xy_accuracy.item(), wh_accuracy.item(), conf_accuracy.item()])


def yolo_accuracy_v3(pred, target, C=20):
    # 假设 pred 和 target 是三个尺度的预测和目标
    large_cell = yolo_accuracy(pred[0], target[0], C)
    medium_cell = yolo_accuracy(pred[1], target[1], C)
    small_cell = yolo_accuracy(pred[2], target[2], C)
    
    return (large_cell + medium_cell + small_cell)/3
   

import math
from torch.amp import autocast, GradScaler


# 定义图像增强策略
transform = transforms.Compose([
    # 将tensor转为PIL以应用transforms
    transforms.ToPILImage(),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(p=0.5),
    # 随机调整亮度、对比度、饱和度、色调
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    # 随机旋转、平移和缩放
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    # 转回tensor
    transforms.ToTensor(),
])

import platform # for CS GPU
if __name__ == '__main__':
    # test_ResUnit() # 1, 32, 416, 416
    # test_ResUnitX() # 1, 64, 208, 208
    # test_Yolov3() # 1, 5, 13, 13

    # 960, 720 -> 30, 23 & 60, 45 & 120, 90


    is_use_autoscale = False

    train_dataset = Comp0249Dataset('data/CamVid', "train", scale=1, transform=None, target_transform=None, version="yolov3")

    if is_use_autoscale:
        if platform.system() == 'Windows':
            train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        if platform.system() == 'Linux':
            train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True)
    else:
        if platform.system() == 'Windows':
            train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4)
        if platform.system() == 'Linux':
            train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=12)


    data = train_dataset[0]
    # print(data[0].shape, data[1].shape)    
    image_size = data[0].shape # (3, h, w)  
    _, h, w = image_size   

    tst_label = [None, None, None]
    tst_label[0] = data[1][0].unsqueeze(0)
    tst_label[1] = data[1][1].unsqueeze(0)
    tst_label[2] = data[1][2].unsqueeze(0)
    model = Yolov3(5, in_channels=3, B=1)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=5
# )

    total_loss = []
    total_acc = []
    num_epochs = 150

    if is_use_autoscale:
        scaler = GradScaler()
    # patience = 3         # 容忍连续多少个 epoch 损失不降
    # min_delta = 0.001    # 损失需要下降的最小改变量
    # best_loss = float('inf')
    # epochs_no_improve = 0

    # test = yolo_loss_funcv3(tst_label, tst_label, w/32, h/32, C=5)
    # print(test)
    # test_acc = yolo_accuracy_v3(tst_label, tst_label, C=5)
    # print(test_acc)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        loss_per_epoch = 0.0
        acc_per_epoch = 0.0

        for images, labels in tqdm(train_loader, desc="Batches"):
            images = images.to(device, dtype=torch.float32)
            for i in range(len(labels)):
                labels[i] = labels[i].to(device, dtype=torch.float)

            optimizer.zero_grad()
            
            if is_use_autoscale:
                with autocast(device_type=str(device)):
                    # 计算预测结果
                    pred = model(images)
                    # pred = labels

                    loss = yolo_loss_funcv3(pred, labels, w/32, h/32, model.B, model.C)

                    # 使用scaler缩放损失并执行反向传播
                    scaler.scale(loss).backward()
                    # 使用scaler更新权重
                    scaler.step(optimizer)
                    # 为下一次迭代更新scaler
                    scaler.update()
            else:
                pred = model(images)
                # pred = labels

                loss = yolo_loss_funcv3(pred, labels, w/32, h/32, model.B, model.C)
                loss.backward()
                optimizer.step()

            loss_per_epoch += loss.item()

            # 使用刚刚定义的 yolo_accuracy 计算存在目标格子的分类准确率
            batch_acc = yolo_accuracy_v3(pred, labels, model.C)
            acc_per_epoch += batch_acc.item()

        loss_per_epoch /= len(train_loader)
        acc_per_epoch /= len(train_loader)
        
        # **在这里调用学习率调整**
        # scheduler.step(loss_per_epoch)  # 这里使用训练损失调整学习率（可改为 val_loss）

        # **打印日志**
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_per_epoch:.4f}, Accuracy: {acc_per_epoch:.4f}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

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
    torch.save(model, 'results/full_model_yolov3_optimize3.pth')
    import json
    data = {
        'loss': total_loss,
        'accuracy': total_acc
    }
    with open('results/data_yolov3_optimize3.json', 'w') as f:
        json.dump(data, f)
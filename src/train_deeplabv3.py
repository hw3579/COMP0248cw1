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

# 定义 ResNet-like 结构
class ConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, is_res=False):
        super(ConvBlock, self).__init__()
        self.is_res = is_res
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels) 
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

        if is_res:
            self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bn_res = nn.BatchNorm2d(out_channels)

        self.mix_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.is_res:
            res = self.conv_res(x)
            res = self.bn_res(res)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.is_res:
            x += res
        x = self.mix_relu(x)
        return x


def test_convblock():
    model = ConvBlock(64, 64, 256)
    # print(model)
    random_tensor = torch.randn(1, 64, 56, 56)
    output = model(random_tensor) # 1, 256, 720, 960
    print(output.shape)


class StageBlock1(nn.Module):
    def __init__(self):
        super(StageBlock1, self).__init__()
        self.convblock1 = ConvBlock(64, 64, 256, stride=1, is_res=True)
        self.convblock2 = ConvBlock(256, 64, 256)
        self.convblock3 = ConvBlock(256, 64, 256)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x


class StageBlock2(nn.Module):
    def __init__(self):
        super(StageBlock2, self).__init__()
        self.convblock1 = ConvBlock(256, 128, 512, stride=2, is_res=True)
        self.convblock2 = ConvBlock(512, 128, 512)
        self.convblock3 = ConvBlock(512, 128, 512)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x
    
class StageBlock3(nn.Module):
    def __init__(self):
        super(StageBlock3, self).__init__()
        self.convblock1 = ConvBlock(512, 256, 1024, stride=2, is_res=True)
        self.convblock2 = ConvBlock(1024, 256, 1024)
        self.convblock3 = ConvBlock(1024, 256, 1024)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x
    
class StageBlock4(nn.Module):
    def __init__(self):
        super(StageBlock4, self).__init__()
        self.convblock1 = ConvBlock(1024, 512, 2048, stride=2, is_res=True)
        self.convblock2 = ConvBlock(2048, 512, 2048)
        self.convblock3 = ConvBlock(2048, 512, 2048)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x


def test_stageblock1():  
    model = StageBlock1()
    # print(model)
    random_tensor = torch.randn(1, 64, 56, 56)
    output = model(random_tensor) # 1, 512, 56, 56
    print(output.shape)

def test_stageblock2_4():
    model = StageBlock4()
    # print(model)
    random_tensor = torch.randn(1, 1024, 14, 14)
    output = model(random_tensor) # 1, 512, 28, 28
    print(output.shape)

class ResNetHead(nn.Module):
    def __init__(self):
        super(ResNetHead, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = StageBlock1()
        self.stage2 = StageBlock2()
        self.stage3 = StageBlock3()
        self.stage4 = StageBlock4()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

def test_resnethead():
    model = ResNetHead()
    # print(model)
    random_tensor = torch.randn(1, 3, 224, 224)
    output = model(random_tensor) # 1, 2048, 30, 23
    print(output.shape)
if __name__ == "__main__":
    pass
    # test_resnethead() # pass in (3, 224, 224) tensor out (2048, 7, 7)

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, mid_channels, out_channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = ConvBlock(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
#         self.conv2 = ConvBlock(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.shortcut = nn.Sequential()
#         if in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
    
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.bn(out)
#         out += self.shortcut(x)
#         return self.relu(out)

# class Backbone(nn.Module):
#     def __init__(self):
#         super(Backbone, self).__init__()
#         self.input = nn.Sequential(
#             ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         self.stage1 = ResidualBlock(64, 64, 256)
#         self.stage2 = ResidualBlock(256, 128, 512)
#         self.stage3 = ResidualBlock(512, 256, 1024)
#         self.stage4 = ResidualBlock(1024, 512, 2048)
    
#     def forward(self, x):
#         x = self.input(x)
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.stage4(x)
#         return x

class ASPP(nn.Module):

    def __init__(self):
        super(ASPP, self).__init__()

        self.aspp1 = self._unit_layer(1, 1, 0)
        self.aspp2 = self._unit_layer(3, 12, 12)
        self.aspp3 = self._unit_layer(3, 24, 24)
        self.aspp4 = self._unit_layer(3, 36, 36)

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 256, 1, stride=1, bias=False),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def _unit_layer(self, kernal_size, dilation, padding, stride=1):
        return nn.Sequential(
            nn.Conv2d(2048, 256, kernal_size, stride, padding, dilation, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
    


    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
    
class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, num_classes, w=480, h=480):
        super(DeepLabV3PlusDecoder, self).__init__()
        self.num_classes = num_classes
        self.w = w
        self.h = h
        # 3x3 卷积（减少通道数 + BN + ReLU）
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 1x1 卷积（将通道数变为类别数）
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv3x3(x)  # 60x60x256 -> 60x60x256
        x = self.classifier(x)  # 60x60x256 -> 60x60xnum_classes
        x = F.interpolate(x, size=(self.h, self.w), mode='bilinear', align_corners=True)
        return x  # 输出分割结果 (480x480xnum_classes)

class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_classes, anchors=None):
        super(YOLOHead, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors if anchors else [[10, 13], [16, 30], [33, 23]]
        self.num_anchors = len(self.anchors)
        
        # 每个框预测5+num_classes个值：x,y,w,h,conf和类别概率
        self.out_channels = self.num_anchors * (5 + num_classes)
        
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)
        
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.LeakyReLU(0.1, inplace=True)
        
        # 最终的预测层
        self.pred = nn.Conv2d(512, self.out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.pred(x)
        
        # 重塑输出为[batch, num_anchors, grid_h, grid_w, 5+num_classes]
        batch_size = x.size(0)
        grid_size = x.size(2)
        
        # 将通道维度重组为检测器所需格式
        prediction = x.view(batch_size, self.num_anchors, 5 + self.num_classes, 
                           grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
        
        return prediction
    
class TotalDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, w=480, h=480):
        super(TotalDeepLabV3Plus, self).__init__()
        self.backbone = ResNetHead()
        self.aspp = ASPP()
        self.decoder = DeepLabV3PlusDecoder(num_classes, w, h)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x)
        return x

def aspp_test():
    model = ASPP()
    # print(model)
    random_tensor = torch.randn(1, 2048, 23, 30)
    output = model(random_tensor) # 1, 512, 30, 23
    print(output.shape)


def deeplabv3plus_test():
    input_tensor = torch.randn(1, 256, 23, 30)
    decoder = DeepLabV3PlusDecoder(num_classes=6, w=input_tensor.shape[2], h=input_tensor.shape[3])
    output = decoder(input_tensor)
    print(output.shape)  # 预期: torch.Size([1, 21, 480, 480])

def totalDeepLabV3Plus_test():
    model = TotalDeepLabV3Plus(num_classes=6, w=960, h=720)
    # print(model)
    random_tensor = torch.randn(1, 3, 720, 960)
    output = model(random_tensor) # 1, 21, 480, 480
    print(output.shape)

if __name__ == "__main__":
    # totalDeepLabV3Plus_test() # pass
    pass

def compute_iou(pred, labels, num_classes=6):
    """
    计算两幅标签图的 IoU（每个类别计算 IoU，最后求平均）。
    
    参数：
        pred (Tensor): 模型预测值 (N, C, H, W)
        labels (Tensor): 真实标签 (N, H, W)，值域 0 ~ num_classes-1
        num_classes (int): 语义分割的类别数（包括背景）
    
    返回：
        iou_dict (dict): 每个类别的 IoU 值
        mean_iou (float): 平均 IoU
    """
    # 将预测结果转换为类别标签
    pred = torch.argmax(pred, dim=1)  # (N, H, W)

    iou_dict = {}
    iou_list = []

    for cls in range(num_classes):
        intersection = ((pred == cls) & (labels == cls)).float().sum()
        union = ((pred == cls) | (labels == cls)).float().sum()

        if union == 0:
            iou = torch.tensor(float('nan'))  # 该类在两幅图中都不存在
        else:
            iou = intersection / union
        
        iou_dict[f'class_{cls}'] = iou#.item()
        if not torch.isnan(iou):
            iou_list.append(iou)

    # 计算平均 IoU（忽略 NaN 类别）
    mean_iou = torch.tensor(iou_list).mean().item()
    
    return iou_dict, mean_iou

def test_compute_iou():
    # 生成两幅随机标签图
    torch.manual_seed(0)
    label1 = torch.randint(0, 6, (1, 480, 480))
    label2 = torch.randint(0, 6, (1, 480, 480))
    
    iou_dict, mean_iou = compute_iou(label1, label2, num_classes=6)
    print(iou_dict)
    print(mean_iou)

import platform
from torch.amp import autocast, GradScaler

if __name__ == "__main__":
    # backbone_test()
    # aspp_test()
    # deeplabv3plus_test()
    #test_compute_iou()


    # start training
    is_use_autoscale = True

    train_dataset = Comp0249Dataset('data/CamVid', "train", scale=1, transform=None, target_transform=None)

    if is_use_autoscale:
        if platform.system() == 'Windows':
            train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        if platform.system() == 'Linux':
            train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=20, pin_memory=True, persistent_workers=True)
    else:
        if platform.system() == 'Windows':
            train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
        if platform.system() == 'Linux':
            train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=10)


    model = TotalDeepLabV3Plus(num_classes=6, w=960, h=720)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_loss = []
    total_acc = []
    num_epochs = 150

    criterion = nn.CrossEntropyLoss()  # 默认会忽略不合法类别



    if is_use_autoscale:
        scaler = GradScaler()

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        loss_per_epoch = 0.0
        acc_per_epoch = 0.0

        for images, labels in tqdm(train_loader, desc="Batches"):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            if is_use_autoscale:
                with autocast(device_type=str(device)):
                    pred = model(images)
                    loss = criterion(pred, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(images)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
            # # 计算预测结果
            # pred = model(images)

            # loss = criterion(pred, labels.long())
            # loss.backward()
            # optimizer.step()

            loss_per_epoch += loss.item()

            # 使用刚刚定义的 yolo_accuracy 计算存在目标格子的分类准确率
            _, batch_acc = compute_iou(pred, labels)
            acc_per_epoch += batch_acc

        loss_per_epoch /= len(train_loader)
        acc_per_epoch /= len(train_loader)
        print(f"Epoch: {epoch}, Loss: {loss_per_epoch:.4f}, Acc: {acc_per_epoch:.4f}")

        total_loss.append(loss_per_epoch)
        total_acc.append(acc_per_epoch)


        if (epoch + 1) % 5 == 0:
            torch.save(model, 'results/deeplabmodelfull3.5.pth')

    # 保存模型及训练指标
    torch.save(model, 'results/deeplabmodelfull3.5.pth')
    import json
    data = {
        'loss': total_loss,
        'accuracy': total_acc
    }
    with open('results/deeplabmodeldata3.5.json', 'w') as f:
        json.dump(data, f)
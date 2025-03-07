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

# 在文件开头的导入部分添加
import datetime

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
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        # 注意这里没有ReLU
        
        if self.is_res:
            identity = self.conv_res(x)
            identity = self.bn_res(identity)
        
        out += identity
        out = self.mix_relu(out)  # 只在残差连接后应用一次ReLU
        
        return out


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

class StageBlock4_2(nn.Module):
    def __init__(self):
        super(StageBlock4_2, self).__init__()
        self.convblock1 = ConvBlock(1024, 512, 2048, stride=3, is_res=True)
        self.convblock2 = ConvBlock(2048, 512, 2048)
        self.convblock3 = ConvBlock(2048, 512, 2048)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

class StageBlockmid(nn.Module):
    def __init__(self):
        super(StageBlockmid, self).__init__()
        self.convblock1 = ConvBlock(2048, 1024, 4096, stride=5, is_res=True)
        self.convblock2 = ConvBlock(4096, 1024, 4096)
        self.convblock3 = ConvBlock(4096, 1024, 4096)

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
        self.stage4_2 = StageBlock4_2()

        # self.extra_conv = nn.Conv2d(512, 2048, kernel_size=3, stride=2, padding=2, dilation=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x_1 = self.stage4(x)
        x_2 = self.stage4_2(x)
        return x_1, x_2

def test_resnethead():
    model = ResNetHead()
    # print(model)
    random_tensor = torch.randn(1, 3, 960, 720)
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

import math
# class YOLOHead(nn.Module): 
#     def __init__(self, in_channels, num_classes, Sx, Sy):
#         super(YOLOHead, self).__init__()
#         self.C = num_classes - 1 # 不包括背景 
#         self.B = 1
#         self.Sx = math.ceil(Sx)
#         self.Sy = math.ceil(Sy)
#         self.adaptive = nn.AdaptiveAvgPool2d((6, 8))  # 强制将输出调整为 (batch, 1024, 3, 3)
#         self.convmid = StageBlockmid()
        
#         # 使用卷积层代替全连接层
#         self.fc_layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(6 * 8 * 4096, 1024),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.5),
#             nn.Linear(1024, 8 * 6 * (self.C + self.B * 5))  # 输出 final_size x final_size x (类别数+边界框数*5)
#         )
    
#     def forward(self, x): # 进来的是2048x20x15
#         # x形状: [batch_size, in_channels, H/32, W/32]
#         # x = self.adaptive(x)
#         x = self.convmid(x)
#         x = F.interpolate(x, size=(6, 8), mode='bilinear', align_corners=True)
#         x = self.fc_layers(x)

#         return x.view(-1, 6, 8, self.C + self.B * 5)  # 输出形状: [batch_size, 8, 6, C + B*5]


class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_classes, Sx, Sy):
        super(YOLOHead, self).__init__()
        self.C = num_classes - 1  # 不包括背景 
        self.B = 1
        self.Sx = math.ceil(Sx)
        self.Sy = math.ceil(Sy)
        
        # 全卷积网络实现
        self.conv_layers = nn.Sequential(
            # 第一层：2048 -> 1024
            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            
            # 第二层：1024 -> 512
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            
            # 第三层：512 -> 256
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            
            # 最后输出层：256 -> (C+B*5)
            nn.Conv2d(256, self.C + self.B * 5, kernel_size=1)
        )
        
    def forward(self, x):  # 进来的是2048x20x15
        # 应用卷积层
        x = self.conv_layers(x)  # 输出 (batch, C+B*5, 20, 15)
        
        # 调整到目标尺寸 (6x8)
        x = F.interpolate(x, size=(6, 8), mode='bilinear', align_corners=True)
        
        # 调整通道顺序：[batch, channels, height, width] -> [batch, height, width, channels]
        return x.permute(0, 2, 3, 1)  # 输出形状: [batch_size, 6, 8, C + B*5]


class TotalDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, w=960, h=720):
        super(TotalDeepLabV3Plus, self).__init__()
        self.backbone = ResNetHead()
        self.aspp = ASPP()
        self.decoder = DeepLabV3PlusDecoder(num_classes, w, h)
        self.yolo_head = YOLOHead(2048, num_classes, w/32, h/32)

    def forward(self, x):
        x, x_yolo = self.backbone(x)
        # batch, 2048, 30, 23

        x_seg = self.aspp(x)    
        x_seg = self.decoder(x_seg)
        x_yolo = self.yolo_head(x_yolo)

        return x_seg, x_yolo

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
    print(output[0].shape, output[1].shape)

if __name__ == "__main__":
    # totalDeepLabV3Plus_test() # pass
    # test_resnethead() # pass
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
from utils import segmentation_to_yolov3_1, yolo_loss, compute_iou_yolo
import os

# 计算类别权重

def calculate_class_weights(dataset, use_cache=True):
    """计算分割任务的类别权重，支持缓存"""
    # 创建缓存目录
    cache_dir = 'data/CamVid/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'class_weights.pth')
    
    # 检查是否存在缓存
    if use_cache and os.path.exists(cache_file):
        print(f"从缓存加载类别权重: {cache_file}")
        class_weights = torch.load(cache_file)
        print(f"加载的类别权重: {class_weights}")
        return class_weights
    
    # 没有缓存，重新计算
    print("计算类别权重...")
    class_counts = {i: 0 for i in range(6)}  # 假设有6个类别（0-5）
    
    for idx in tqdm(range(len(dataset)), desc="分析类别分布"):
        _, [label, _] = dataset[idx]
        unique_classes, counts = torch.unique(label, return_counts=True)
        
        for cls, count in zip(unique_classes.tolist(), counts.tolist()):
            if cls in class_counts:
                class_counts[cls] += count
    
    print(f"类别分布: {class_counts}")
    
    # 计算权重（少数类获得更高权重）
    total_pixels = sum(class_counts.values())
    class_weights = torch.ones(6)  # 假设6个类别
    for cls, count in class_counts.items():
        if count > 0:
            # 使用反比例权重并规范化
            class_weights[cls] = total_pixels / (len(class_counts) * count)
    
    # 归一化权重
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    print(f"类别权重: {class_weights}")
    
    # 保存缓存
    if use_cache:
        torch.save(class_weights, cache_file)
        print(f"已保存类别权重缓存: {cache_file}")
    
    return class_weights

# 在文件顶部导入区添加这个类

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss 实现，用于语义分割
        
        参数:
            alpha: 类别权重 (tensor)，形状为 (C,)
            gamma: 聚焦参数，减少易分类样本的贡献
            reduction: 'none'|'mean'|'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # inputs形状: [N, C, H, W]
        # targets形状: [N, H, W]
        
        # 计算交叉熵损失 (无reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # 获取概率预测
        inputs_softmax = F.softmax(inputs, dim=1)
        
        # 创建one-hot编码的targets
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [N, H, W, C] -> [N, C, H, W]
        
        # 计算目标类别的预测概率
        pt = (inputs_softmax * targets_one_hot).sum(dim=1)  # [N, H, W]
        
        # 计算Focal Loss权重因子
        focal_weight = (1 - pt) ** self.gamma
        
        # 应用Focal Loss权重
        loss = focal_weight * ce_loss
        
        # 应用reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

if __name__ == "__main__":
    # backbone_test()
    # aspp_test()
    # deeplabv3plus_test()
    #test_compute_iou()


    # 在if __name__ == "__main__":之后、训练循环之前添加
    # 检查并清除可能存在的退出标记文件
    exit_file = "exit_training.txt"
    if os.path.exists(exit_file):
        os.remove(exit_file)
    print(f"训练过程中，创建文件 '{exit_file}' 将保存当前模型并退出训练")



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

    # start training
    is_use_autoscale = True

    train_dataset = Comp0249Dataset('data/CamVid', "train", scale=1, transform=None, target_transform=None)

    if is_use_autoscale:
        if platform.system() == 'Windows':
            train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        if platform.system() == 'Linux':
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    else:
        if platform.system() == 'Windows':
            train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
        if platform.system() == 'Linux':
            train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=10)


    # model = TotalDeepLabV3Plus(num_classes=6, w=960, h=720)
    model = torch.load('results/deeplabmodelfullfinal_interrupted.pth', weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 稍微提高初始学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    total_loss = []
    total_acc = []
    num_epochs = 500

    # criterion = nn.CrossEntropyLoss()  # 默认会忽略不合法类别
    class_weights = calculate_class_weights(train_dataset)
    class_weights = class_weights.to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    lambda_yolo = 0

    if is_use_autoscale:
        scaler = GradScaler()

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        loss_per_epoch = 0.0
        seg_loss_per_epoch = 0.0
        yolo_loss_per_epoch = 0.0
        acc_per_epoch = 0.0
        yolo_acc_per_epoch = 0.0  # 新增YOLO准确率统计变量

        for images, labels in tqdm(train_loader, desc="Batches"):
            images = images.to(device, dtype=torch.float32)
            labels_segment = labels[0].to(device, dtype=torch.long)
            labels_yolo = labels[1].to(device, dtype=torch.float32)

            optimizer.zero_grad()
            if is_use_autoscale:
                with autocast(device_type=str(device)):
                    # 在训练循环中使用修改后的损失函数
                    pred, pred_yolo = model(images)
                    seg_loss = criterion(pred, labels_segment)
                    yolo_loss_val = yolo_loss(pred_yolo, labels_yolo, 8, 6, 1, 5, gamma=2.0, alpha=0.25)

                    lambda_yolo = seg_loss_per_epoch / (yolo_loss_per_epoch + 1e-8)

                    batch_total_loss = seg_loss + yolo_loss_val * lambda_yolo
                
                # 使用正确的损失进行缩放和反向传播
                scaler.scale(batch_total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred, pred_yolo = model(images)
                seg_loss = criterion(pred, labels_segment)
                yolo_loss_val = yolo_loss(pred_yolo, labels_yolo, 8, 6, 1, 5)

                lambda_yolo = seg_loss_per_epoch / (yolo_loss_per_epoch + 1e-8)

                batch_total_loss = seg_loss + yolo_loss_val * lambda_yolo
                batch_total_loss.backward()
                optimizer.step()

            # 累积各类损失
            loss_per_epoch += batch_total_loss.item()
            seg_loss_per_epoch += seg_loss.item()
            yolo_loss_per_epoch += yolo_loss_val.item()

            # 计算准确率
            _, batch_acc = compute_iou(pred, labels_segment)
            acc_per_epoch += batch_acc

                    # 添加：计算YOLO准确率
            with torch.no_grad():
                batch_yolo_iou, _ = compute_iou_yolo(pred_yolo, labels_yolo)
                yolo_acc_per_epoch += batch_yolo_iou

        # 计算每个epoch的平均损失和准确率
        seg_loss_per_epoch /= len(train_loader)
        yolo_loss_per_epoch /= len(train_loader)
        loss_per_epoch /= len(train_loader)
        acc_per_epoch /= len(train_loader)
        yolo_acc_per_epoch /= len(train_loader)  # 计算YOLO平均准确率

        
        print(f"segloss:{seg_loss_per_epoch:.4f}, yololoss:{yolo_loss_per_epoch:.4f}")
        print(f"Epoch: {epoch}, Loss: {loss_per_epoch:.4f}, Acc: {acc_per_epoch:.4f}, YOLO Acc: {yolo_acc_per_epoch:.4f}")

        total_loss.append(loss_per_epoch)
        total_acc.append(acc_per_epoch)
        

        # 添加：记录YOLO准确率
        if 'total_yolo_acc' not in locals():
            total_yolo_acc = []
        total_yolo_acc.append(yolo_acc_per_epoch)


        # 学习率调整
        scheduler.step(loss_per_epoch)


        # early stopping
        patience = 10  # 连续多少个epoch没改善就停止
        min_delta = 1e-4  # 改善的最小阈值
        min_epochs_before_earlystop = 2000  # 至少训练这么多epoch才开始检查早停

        # 初始化早停所需变量
        if epoch == 0:
            best_loss = loss_per_epoch
            best_epoch = 0
            counter = 0
        # 判断是否有改善
        elif loss_per_epoch < best_loss - min_delta:
            best_loss = loss_per_epoch
            best_epoch = epoch
            counter = 0
            # 保存最佳模型
            torch.save(model, 'results/deeplabmodelfullfinal.pth')
        else:
            # 只有在达到最小训练轮数后才增加早停计数器
            if epoch >= min_epochs_before_earlystop:
                counter += 1
                print(f"Early stopping counter: {counter}/{patience} (active after {min_epochs_before_earlystop} epochs)")
            else:
                print(f"Early stopping not active yet, will activate after {min_epochs_before_earlystop} epochs")
            
        # 如果连续patience个epoch没有改善且已经过了最小训练轮数，停止训练
        if counter >= patience and epoch >= min_epochs_before_earlystop:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with loss {best_loss:.4f}")
            break




        # 修改检测退出信号的代码段
        if os.path.exists(exit_file):
            print("\n检测到退出信号，正在保存模型和训练数据...")
            
            # 生成时间戳
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存当前模型（使用特殊名称和时间戳以避免覆盖）
            model_filename = f'results/deeplabmodelfullfinal_interrupted.pth'
            torch.save(model, model_filename)
            
            import json
            # 在检测退出信号的代码段中修改data字典
            data = {
                'loss': total_loss,
                'accuracy': total_acc,
                'yolo_accuracy': total_yolo_acc,  # 添加YOLO准确率记录
                'last_epoch': epoch,
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'interrupt_time': timestamp
                }
            
            # 保存训练数据，同样使用时间戳
            data_filename = f'results/deeplabmodeldatafinal_interrupted_{timestamp}.json'
            with open(data_filename, 'w') as f:
                json.dump(data, f)
            
            # 删除触发文件
            os.remove(exit_file)
            print(f"保存完成，文件已保存为:\n- {model_filename}\n- {data_filename}")
            print("训练已退出。")
            break

    # 保存模型及训练指标
    # torch.save(model, 'results/deeplabmodelfull3.5.pth')
    # import json
    # data = {
    #     'loss': total_loss,
    #     'accuracy': total_acc
    # }
    # with open('results/deeplabmodeldata3.5.json', 'w') as f:
    #     json.dump(data, f)
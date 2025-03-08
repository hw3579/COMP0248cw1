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
import math

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
    print(output[0].shape, output[1].shape)
if __name__ == "__main__":
    # test_resnethead() # pass in (3, 224, 224) tensor out (2048, 7, 7)

    pass

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
    def __init__(self, in_channels, num_classes, Sx=None, Sy=None):
        super(YOLOHead, self).__init__()
        self.C = num_classes - 1  # 不包括背景 
        self.B = 1
        
        # 保持输入尺寸不变的卷积网络
        self.conv1 = nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.act1 = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.act2 = nn.LeakyReLU(0.1)
        
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.1)
        
        # 输出层：保持空间维度不变，只改变通道数
        self.conv_out = nn.Conv2d(256, self.C + self.B * 5, kernel_size=1)
        
    def forward(self, x):  # 输入 2048×20×15
        # 应用卷积层，逐层处理，保持空间维度不变
        x = self.conv1(x)  # 输出 1024×20×15
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.conv2(x)  # 输出 512×20×15
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.conv3(x)  # 输出 256×20×15
        x = self.bn3(x)
        x = self.act3(x)
        
        x = self.conv_out(x)  # 输出 (C+B*5)×20×15
                
        # 调整通道顺序：[batch, channels, height, width] -> [batch, height, width, channels]
        return x.permute(0, 2, 3, 1)  # 输出形状: [batch_size, 20, 15, C+B*5]


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
    # backbone_test()
    # aspp_test()
    # deeplabv3plus_test()
    totalDeepLabV3Plus_test() # pass
    # test_resnethead() # pass
    pass
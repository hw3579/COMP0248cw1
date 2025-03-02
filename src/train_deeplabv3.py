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

class Bottleneck(nn.Module):
    expansion = 4  # 通道扩展比例

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 下采样用于调整通道数
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # 维度匹配

        out += identity
        out = self.relu(out)

        return out

class Backbone(nn.Module):
    def __init__(self, block, layers=[3, 4, 6, 3], num_classes=1000):
        super(Backbone, self).__init__()
        self.in_channels = 64  # 初始输入通道数

        #Layer 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #Layer 2
        # ResNet 四个层（每层包含多个 Bottleneck Block）
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化 + FC 分类
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        self._initialize_weights()



    def _make_layer(self, block, mid_channels, blocks, stride=1):
        """ 构建 ResNet 的层 """
        downsample = None
        if stride != 1 or self.in_channels != mid_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, mid_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, mid_channels, stride, downsample))
        self.in_channels = mid_channels * block.expansion  # 更新输入通道数

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, mid_channels))

        return nn.Sequential(*layers)
    

    def _initialize_weights(self):
        """ 初始化权重 """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv1(x)  # 7x7 卷积
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 3x3 最大池化

        x = self.layer1(x)  # Layer1
        x = self.layer2(x)  # Layer2
        x = self.layer3(x)  # Layer3
        x = self.layer4(x)  # Layer4


        return x
    

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


class TotalDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, w=480, h=480):
        super(TotalDeepLabV3Plus, self).__init__()
        self.backbone = Backbone(Bottleneck)
        self.aspp = ASPP()
        self.decoder = DeepLabV3PlusDecoder(num_classes, w, h)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x)
        return x


def backbone_test():
    model = Backbone(Bottleneck)
    # print(model)
    random_tensor = torch.randn(1, 3, 720, 960)
    output = model(random_tensor) # 1, 2048, 30, 23
    print(output.shape)

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


if __name__ == "__main__":
    # backbone_test()
    # aspp_test()
    # deeplabv3plus_test()
    #test_compute_iou()


    # start training
    train_dataset = Comp0249Dataset('data/CamVid', "train", scale=1)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

    model = TotalDeepLabV3Plus(num_classes=6, w=960, h=720)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_loss = []
    total_acc = []
    num_epochs = 150

    criterion = nn.CrossEntropyLoss()  # 默认会忽略不合法类别




    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        loss_per_epoch = 0.0
        acc_per_epoch = 0.0

        for images, labels in tqdm(train_loader, desc="Batches"):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()

            # 计算预测结果
            pred = model(images)

            loss = criterion(pred, labels.long())
            loss.backward()
            optimizer.step()

            loss_per_epoch += loss.item()

            # 使用刚刚定义的 yolo_accuracy 计算存在目标格子的分类准确率
            _, batch_acc = compute_iou(pred, labels)
            acc_per_epoch += batch_acc

        loss_per_epoch /= len(train_loader)
        acc_per_epoch /= len(train_loader)
        print(f"Epoch: {epoch}, Loss: {loss_per_epoch:.4f}, Acc: {acc_per_epoch:.4f}")

        total_loss.append(loss_per_epoch)
        total_acc.append(acc_per_epoch)

    # 保存模型及训练指标
    torch.save(model, 'results/deeplabmodelfull.pth')
    import json
    data = {
        'loss': total_loss,
        'accuracy': total_acc
    }
    with open('results/deeplabmodeldata.json', 'w') as f:
        json.dump(data, f)
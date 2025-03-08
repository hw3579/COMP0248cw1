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
from model import ResNetHead, ASPP, DeepLabV3PlusDecoder, ConvBlock, StageBlock1, StageBlock2, StageBlock3, StageBlock4, YOLOHead, TotalDeepLabV3Plus, StageBlockmid, StageBlock4_2
import platform
from torch.amp import autocast, GradScaler
from utils import segmentation_to_yolov3_1, yolo_loss, compute_iou_yolo
import os

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
            train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0, pin_memory=True)
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
    # from torchsummary import summary
    # print(summary(model, (3, 720, 960)))

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 稍微提高初始学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    total_loss = []
    total_acc = []
    num_epochs = 500

    criterion = nn.CrossEntropyLoss()  # 默认会忽略不合法类别
    # class_weights = calculate_class_weights(train_dataset)
    # class_weights = class_weights.to(device)
    # criterion = FocalLoss(alpha=class_weights, gamma=2.0)

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


                    # pred, pred_yolo = images, labels_yolo

                    # b, h, w = labels_segment.size()
                    # num_classes = 6
                    # pred_segment = torch.zeros(b, num_classes, h, w, device=device)
                    # # 为每个标签在对应位置设置高值(10.0)，模拟softmax前的logits
                    # for i in range(num_classes):
                    #     pred_segment[:, i, :, :] = (labels_segment == i).float() * 10.0

                    # pred = pred_segment




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
                batch_yolo_iou_, batch_yolo_iou = compute_iou_yolo(pred_yolo, labels_yolo)
                yolo_acc_per_epoch += batch_yolo_iou

        # 计算每个epoch的平均损失和准确率
        seg_loss_per_epoch /= len(train_loader)
        yolo_loss_per_epoch /= len(train_loader)
        loss_per_epoch /= len(train_loader)
        acc_per_epoch /= len(train_loader)
        yolo_acc_per_epoch /= len(train_loader)  # 计算YOLO平均准确率

        
        print(f"segloss:{seg_loss_per_epoch:.4f}, yololoss:{yolo_loss_per_epoch:.4f}")
        print(f"Epoch: {epoch}, Loss: {loss_per_epoch:.4f}, Acc: {acc_per_epoch:.4f}, YOLO Class Acc: {yolo_acc_per_epoch:.4f}")

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
        min_epochs_before_earlystop = 75  # 至少训练这么多epoch才开始检查早停

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
            torch.save(model, 'results/deeplabmodelfullfinal_interrupted.pth')
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
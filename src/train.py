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



# Build Dert manually 

class TotalModel(nn.Module):
    def __init__(self, in_channels, num_classes=5):
        """
        in_channels: 输入图像的通道数
        num_classes: 实际类别数量，不含背景，默认5
        """
        super(TotalModel, self).__init__()
        
        # ---- 1. CNN部分 ----
        # 这里最后输出通道数改为 32，方便与后续Transformer对接
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(64, 128, 3, padding=1),
            # nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU()
        )
        # 1×1卷积，若想对特征进行再次通道变换或压缩可保留，否则可删
        self.conv1x1 = nn.Conv2d(32, 32, 1)


        # ---- 2. 位置嵌入与查询向量 ----
        # 注意：这里用 1 × (72*96) × 32 来做演示，如果分辨率很大则数值也应同步调整
        self.position_embedding = nn.Parameter(torch.randn(1, 691200, 32))
        self.query_embed = nn.Parameter(torch.randn(691200, 32))

        # ---- 3. Transformer部分 ----
        # d_model=32 与 CNN 输出对齐
        self.transformer = nn.Transformer(
            d_model=32, 
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            batch_first=True
        )
        self.segmentation_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # 假设输入 x 尺寸为 (B, in_channels, H, W)
        B, _, H, W = x.shape
        
        # CNN 部分
        x = self.cnn(x)
        x = self.conv1x1(x)  # (B, 32, H, W)
        
        # 将二维特征图展平为序列 (B, H×W, 32)
        x = x.view(B, 32, H * W).permute(0, 2, 1)
        
        # 添加位置嵌入（注意这里假设 H*W 与预设的 self.seq_length 匹配，
        # 若实际尺寸不同，可相应调整或使用插值）
        n_pos = H * W
        if n_pos != self.position_embedding.shape[1]:
            pos_embed = F.interpolate(self.position_embedding.transpose(1,2), size=n_pos, mode='linear').transpose(1,2)
        else:
            pos_embed = self.position_embedding
        x = x + pos_embed
        
        # 构造查询向量（与特征图大小一致）

        # 假设 x 已经计算好了，且当前序列长度为 n_pos = H * W
        if n_pos != self.query_embed.shape[0]:
            queries = F.interpolate(
                self.query_embed.unsqueeze(0).transpose(1,2),
                size=n_pos,
                mode='linear'
            ).transpose(1,2)
            queries = queries.repeat(B, 1, 1)
        else:
            queries = self.query_embed.unsqueeze(0).repeat(B, 1, 1)

        
        # Transformer 部分，输入 src 为 CNN 特征，tgt 为查询向量
        x = self.transformer(src=x, tgt=queries)  # (B, H×W, 32)
        
        # 恢复成二维特征图 (B, 32, H, W)
        x = x.permute(0, 2, 1).view(B, 32, H, W)
        
        # segmentation head 输出各像素类别预测
        out = self.segmentation_head(x)  # (B, num_classes, H, W)
        return out


def segmentation_loss(pred, target):
    # pred: (B, num_classes, H, W)
    # target: (B, H, W), 每个像素是对应类别的整数索引
    return F.cross_entropy(pred, target.long())

def early_stopping(loss, patience=5):
    if len(loss) < patience:
        return False
    for i in range(1, patience):
        if loss[-i] < loss[-i-1]:
            return False
    return True

if __name__ == "__main__":
    model = TotalModel(3, 6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    train_dataset = Comp0249Dataset('data/CamVid', "train", transform=None, target_transform=None)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)         

    total_loss = []
    total_acc = []
    num_epochs = 10
    for epoch in tqdm(range(num_epochs)):

        loss_per_epoch = 0
        acc_per_epoch = 0

        for images, labels in tqdm(train_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()

            # 计算预测结果
            pred = model(images)

            loss = segmentation_loss(pred, labels)
            loss.backward()
            optimizer.step()

            loss_per_epoch += loss.item()

            pred_labels = torch.argmax(pred, dim=1)
            batch_acc = (pred_labels == labels).float().mean()

            acc_per_epoch += batch_acc.item()



        loss_per_epoch /= len(train_loader)
        acc_per_epoch /= len(train_loader)
        print(f"epcho: {epoch}, loss: {loss_per_epoch}, acc: {acc_per_epoch}")

        total_loss.append(loss_per_epoch)
        total_acc.append(acc_per_epoch)

        if early_stopping(total_loss):
            break

    #save model
    torch.save(model.state_dict(), 'results/model.pth')
    torch.save(model, 'results/full_model.pth')

    #save the loss and accuracy as json
    import json
    data = {
        'loss': total_loss,
        'accuracy': total_acc
    }
    with open('results/data.json', 'w') as f:
        json.dump(data, f)
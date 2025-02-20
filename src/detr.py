import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
from dataloader import Comp0249Dataset
import torch.optim as optim
from torch.utils.data import DataLoader

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class CNNBackbone(nn.Module):
    def __init__(self, num_channels=3, hidden_dim=256):
        super(CNNBackbone, self).__init__()
        self.conv = nn.Conv2d(num_channels, hidden_dim, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_height=224, max_width=224):
        super(PositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(max_height, d_model // 2)
        self.col_embed = nn.Embedding(max_width, d_model // 2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        # x: [batch, channels, height, width]
        h, w = x.shape[2], x.shape[3]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)  # [w, d_model//2]
        y_emb = self.row_embed(j)  # [h, d_model//2]
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1)
        ], dim=-1)  # [h, w, d_model]
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = src + pos
        src2 = self.self_attn(q, k, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, pos=None, query_pos=None):
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt + query_pos, key=memory + pos, value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, tgt, memory, pos, query_pos, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
        return output

class Transformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, num_queries=100):
        super(Transformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.num_queries = num_queries

    def forward(self, src, mask, query_embed, pos_embed):
        # src: [batch, d_model, H, W]，先将空间维度展平并调整维度顺序
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)       # [HW, batch, d_model]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # [HW, batch, d_model]
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # 准备查询嵌入，shape 为 [num_queries, batch, d_model]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, memory, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(0, 1)  # 输出 shape: [batch, num_queries, d_model]


class DETR(nn.Module):
    def __init__(self, num_classes, num_queries, d_model=256, hidden_dim=256):
        super(DETR, self).__init__()
        self.backbone = CNNBackbone(num_channels=3, hidden_dim=hidden_dim)
        self.conv = nn.Conv2d(hidden_dim, d_model, kernel_size=1)  # 将特征映射到 d_model 维度
        self.position_embedding = PositionalEncoding(d_model)
        self.transformer = Transformer(d_model=d_model, nhead=8,
                                       num_encoder_layers=6, num_decoder_layers=6,
                                       dim_feedforward=2048, dropout=0.1,
                                       num_queries=num_queries)
        self.query_embed = nn.Embedding(num_queries, d_model)
        # 分类头：输出类别数 + 1（代表 no-object）
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        # 边界框回归头
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        # 新增分割头：将 backbone 特征映射转换为分割图 (保留原始分辨率或通过插值调整)
        self.segmentation_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, data):
        # data 为 (image, label) 元组
        image, label = data
        features = self.backbone(image)
        # 得到分割输出，并上采样到与 label 一致的尺寸
        seg_logits = self.segmentation_head(features)
        seg_logits = F.interpolate(seg_logits, size=label.shape[-2:], mode='bilinear', align_corners=False)
        src = self.conv(features)
        pos = self.position_embedding(src)
        mask = None  # 如有需要，可自行定义 mask
        hs = self.transformer(src, mask, self.query_embed.weight, pos)
        outputs_class = self.class_embed(hs)
        outputs_bbox = self.bbox_embed(hs).sigmoid()
        # 返回分割 logits 用于计算交叉熵损失，以及检测分支的输出
        return {'pred_logits': seg_logits, 'pred_boxes': outputs_bbox, 'labels': label}

loss_fn = nn.CrossEntropyLoss()

def segmentation_loss(pred, target): 
    # pred['pred_logits'] 尺寸：[batch, num_classes, H, W] 
    #  target 尺寸：[batch, H, W]（每个像素的类别标签） 
    return loss_fn(pred['pred_logits'], target)

def early_stopping(loss_list, patience=3, min_delta=0.0):
    # 当累计的 epoch 数还不足 patience+1 时，不触发早停
    if len(loss_list) < patience + 1:
        return False
    # 前面所有 epoch 中的最佳损失
    best_loss = min(loss_list[:-patience])
    # 如果最近 patience 个 epoch 内的损失均没有比 best_loss 降低 min_delta，则触发早停
    if all(loss >= best_loss - min_delta for loss in loss_list[-patience:]):
        return True
    return False

if __name__ == "__main__":
    train_dataset = Comp0249Dataset('data/CamVid', "train", transform=None, target_transform=None)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    model = DETR(num_classes=6, num_queries=100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    total_loss = []
    total_acc = []
    num_epochs = 15
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        loss_per_epoch = 0.0
        acc_per_epoch = 0.0
        model.train()
        
        for images, labels in tqdm(train_loader, desc="Batches", leave=False):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            # 此处输入为 (image, label) 元组，模型内部只用 image 进行前向传播
            pred = model((images, labels))
            loss = segmentation_loss(pred, labels)
            loss.backward()
            optimizer.step()
            
            loss_per_epoch += loss.item()
            
            # 假设模型输出 pred_logits 后经过 argmax 得到 [batch, H, W] 的预测分割结果
            pred_labels = torch.argmax(pred['pred_logits'], dim=1)
            batch_acc = (pred_labels == labels).float().mean()
            acc_per_epoch += batch_acc.item()
        
        loss_per_epoch /= len(train_loader)
        acc_per_epoch /= len(train_loader)
        
        print(f"epoch: {epoch}, loss: {loss_per_epoch:.4f}, acc: {acc_per_epoch:.4f}")
        total_loss.append(loss_per_epoch)
        total_acc.append(acc_per_epoch)
        
        if early_stopping(total_loss, patience=3, min_delta=0.001):
            print("Early stopping triggered. Training terminated.")
            break

    # save model
    torch.save(model.state_dict(), 'results/model2.pth')
    torch.save(model, 'results/full_model2.pth')

    # save the loss and accuracy as json
    import json
    data = {
        'loss': total_loss,
        'accuracy': total_acc
    }
    with open('results/data2.json', 'w') as f:
        json.dump(data, f)
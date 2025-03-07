
import torch
import numpy as np
import cv2


# from dataloader import Comp0249Dataset

def draw_the_box(image, pred_label):
    '''
    input: image - the image to draw the box on
            pred_label - the predicted label (RGB image) (H, W, C) PIL image
    output: image - the image with the box drawn on it (H, W)
    '''
    H, W, _ = image.shape
    box_mask = np.zeros((5, H, W), dtype=np.uint8)

    for i in range(5):
        # 得到当前类别的掩码，类型为 bool
        mask = (pred_label == i)
        # 如果该类别没有像素，则跳过
        if np.count_nonzero(mask) == 0:
            continue

        # 获得该类别像素的行和列索引
        rows, cols = np.where(mask)
        top = np.min(rows)
        bottom = np.max(rows)
        left = np.min(cols)
        right = np.max(cols)

        # 保证 image 内存连续
        image = np.ascontiguousarray(image)
        # 绘制矩形框
        cv2.rectangle(image, (left, top), (right, bottom), (0, 25*i, 0), 1)

    return image



def segmentation_to_yolov3_1(label, Sx, Sy, num_classes=20, B=2, scale=1): # 重载
    """
    将分割标签 (H, W) 转换为 YOLO 格式标签 (Sy, Sx, num_classes + B*5)
    参数：
        label: torch.Tensor, 分割标签，尺寸为 (H, W)，像素值 0 表示背景，其余值对应类别编号（建议连续）
        Sx: YOLO 网格水平方向尺寸（宽度方向网格数）
        Sy: YOLO 网格垂直方向尺寸（高度方向网格数）
        num_classes: 类别数
        B: 每个网格预测的边框数，通常这里只用第一个框进行标注，其余保持0
    返回：
        yolo_label: torch.Tensor, 尺寸 (Sy, Sx, num_classes + B*5)
                    格式为：前 num_classes 维：目标类别；
                            后面 B*5 维，每个边框组成 [cx, cy, w, h, confidence]
    """
    Sx = math.ceil(Sx)
    Sy = math.ceil(Sy)
    # 将 label 转为 numpy 数组
    label_np = label.cpu().numpy().astype(np.uint8)
    if len(label_np.shape) == 3:
        label_np = label_np[0]
    H, W = label_np.shape
    yolo_label = np.zeros((Sy, Sx, B * 5 + num_classes), dtype=np.float32)

    # 使用 OpenCV 连通域分析（假设背景为 0）
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(label_np, connectivity=8)
    stats = np.concatenate([stats, centroids], axis=1)  # 将中心点坐标拼接到 stats 中
    # 过滤掉面积小于 1e3 的区域（背景不计入）
    filtered_stats = stats[stats[:, -3] >= (1e3//scale)].astype(np.int32)
    # 遍历除背景外的每个目标
    for i in range(1, len(filtered_stats)):
        x, y, w, h, area, cx, cy = filtered_stats[i]  # 边框左上角坐标、宽、高、面积       # 边界框中心点 (浮点)
        # 归一化边框参数（相对于整幅图像尺寸）
        norm_cx = cx / W
        norm_cy = cy / H
        norm_w = w / W
        norm_h = h / H

        # 获取目标类别 —— 这里采用区域内出现频率最高的类别编号
        region = label_np[y:y+h, x:x+w]
        # 注意：背景为0，不计入类别统计
        region_flat = region[region > 0]
        if len(region_flat) == 0:
            continue
        class_id = np.bincount(region_flat).argmax()

        # 确定目标中心所在的网格单元
        grid_x = int(norm_cx * Sx)
        grid_y = int(norm_cy * Sy)
        grid_x = min(grid_x, Sx - 1)
        grid_y = min(grid_y, Sy - 1)

        # 设置该网格单元的类别信息与边框信息（这里只填充第一个边框，后续边框保持0）
        yolo_label[grid_y, grid_x, class_id - 1] = 1
        # 在segmentation_to_yolov3_1函数中添加相对偏移计算
        # 计算相对于网格的偏移量
        rel_cx = norm_cx * Sx - grid_x  # x偏移，范围[0,1]
        rel_cy = norm_cy * Sy - grid_y  # y偏移，范围[0,1]
        
        # 存储相对坐标
        yolo_label[grid_y, grid_x, num_classes:num_classes+5] = np.array([rel_cx, rel_cy, norm_w, norm_h, 1.0], dtype=np.float32)

    # 转换回 torch.Tensor
    return torch.from_numpy(yolo_label)

import math


from torch.nn import functional as F

def yolo_loss(predictions, targets, Sx, Sy, B=1, C=5, lambda_coord=5, lambda_noobj=0.5, gamma=2.0, alpha=0.25, imbalanced_mode=False):
    """修正后的YOLO损失函数计算，支持不平衡数据处理模式
    
    参数:
        predictions: 模型预测结果
        targets: 目标标签
        Sx, Sy: YOLO网格尺寸
        B: 每个网格的边界框数量
        C: 类别数量
        lambda_coord: 坐标损失权重
        lambda_noobj: 无目标网格的置信度损失权重
        gamma: Focal Loss的gamma参数
        alpha: Focal Loss的alpha参数
        imbalanced_mode: 是否启用不平衡数据处理模式
    """
    batch = predictions.shape[0]  # 获取batch大小
    
    # 获取预测和目标
    pred_class = predictions[..., :C]
    target_class = targets[..., :C]
    
    # 创建有目标的掩码
    obj_mask = (targets[..., C+4] > 0.5)
    noobj_mask = ~obj_mask
    
    # 如果启用不平衡处理模式，计算类别权重
    if imbalanced_mode:
        # 统计每个类别的目标数量
        class_counts = []
        for c in range(C):
            # 统计该类别的目标数量
            class_mask = target_class[..., c] > 0.5
            class_count = (class_mask & obj_mask).sum().float() + 1e-6  # 防止除零
            class_counts.append(class_count)
        
        # 将类别数量转换为权重
        class_counts = torch.tensor(class_counts, device=predictions.device)
        total_count = class_counts.sum()
        
        # 类别权重与类别频率成反比
        class_weights = total_count / (C * class_counts)
        
        # 归一化类别权重
        class_weights = class_weights / class_weights.sum() * C
    else:
        class_weights = torch.ones(C, device=predictions.device)
    
    # 1. 只对有目标的网格计算类别损失(使用Focal Loss)
    bce_loss = F.mse_loss(pred_class, target_class, reduction='none')
    pt = torch.exp(-bce_loss)
    
    # 区分正负样本的alpha
    pos_mask = (target_class > 0.5)
    neg_mask = ~pos_mask
    
    if imbalanced_mode:
        # 对不同类别使用不同的alpha权重
        alpha_weight = torch.zeros_like(target_class)
        for c in range(C):
            # 对于正样本，使用类别权重调整alpha
            alpha_weight[..., c] = pos_mask[..., c] * (alpha * class_weights[c]) + neg_mask[..., c] * (1-alpha)
    else:
        alpha_weight = pos_mask * alpha + neg_mask * (1-alpha)
    
    # 应用Focal Loss权重
    focal_weight = alpha_weight * (1 - pt) ** gamma
    
    # 只对有目标的网格计算类别损失
    obj_expanded = obj_mask.unsqueeze(-1).expand_as(pred_class)
    focal_loss = focal_weight * bce_loss * obj_expanded
    class_loss = focal_loss.sum()
    
    # 2. 只对有目标的网格计算坐标损失
    pred_xy = predictions[..., C:C+2]
    pred_wh = predictions[..., C+2:C+4]
    target_xy = targets[..., C:C+2]
    target_wh = targets[..., C+2:C+4]
    
    # 对有目标的网格应用掩码
    obj_expanded_xy = obj_mask.unsqueeze(-1).expand_as(pred_xy)
    obj_expanded_wh = obj_mask.unsqueeze(-1).expand_as(pred_wh)
    
    # 计算坐标损失
    xy_loss = F.mse_loss(
        pred_xy * obj_expanded_xy, 
        target_xy * obj_expanded_xy, 
        reduction='mean'
    )
    
    # 宽高损失使用平方根变换
    wh_loss = F.mse_loss(
        torch.sqrt(torch.abs(pred_wh) + 1e-6) * obj_expanded_wh, 
        torch.sqrt(torch.abs(target_wh) + 1e-6) * obj_expanded_wh, 
        reduction='mean'
    )
    
    coord_loss = lambda_coord * (xy_loss + wh_loss)
    
    # 3. 置信度损失计算
    if imbalanced_mode and obj_mask.sum() > 0:
        # 获取每个目标的类别索引
        target_class_idx = torch.argmax(target_class, dim=-1)
        
        # 创建置信度权重矩阵
        conf_weights = torch.ones_like(targets[..., C+4])
        
        # 对有目标的网格，根据类别频率设置权重
        for i in range(batch):
            for y in range(Sy):
                for x in range(Sx):
                    if obj_mask[i, y, x]:
                        cls_idx = target_class_idx[i, y, x].item()
                        conf_weights[i, y, x] = class_weights[cls_idx]
        
        obj_conf_loss = F.mse_loss(
            predictions[..., C+4][obj_mask], 
            targets[..., C+4][obj_mask], 
            reduction='mean'
        ) * conf_weights[obj_mask].mean()
    else:
        obj_conf_loss = F.mse_loss(
            predictions[..., C+4][obj_mask], 
            targets[..., C+4][obj_mask], 
            reduction='mean'
        )
    
    # 调整无目标损失的权重
    lambda_noobj_adjusted = lambda_noobj
    if imbalanced_mode:
        # 如果有目标的网格很少，进一步降低无目标损失权重
        obj_ratio = obj_mask.sum().float() / obj_mask.numel()
        if obj_ratio < 0.01:  # 少于1%
            lambda_noobj_adjusted = lambda_noobj * 0.5
    
    noobj_conf_loss = lambda_noobj_adjusted * F.mse_loss(
        predictions[..., C+4][noobj_mask], 
        targets[..., C+4][noobj_mask], 
        reduction='mean'
    )
    
    # 4. 归一化损失
    total_loss = class_loss + coord_loss + obj_conf_loss + noobj_conf_loss
    
    if imbalanced_mode:
        # 不平衡模式下不进行额外归一化
        return total_loss
    else:
        # 原始归一化方式
        num_obj = obj_mask.sum().float() + 1e-6
        return total_loss / (batch * num_obj)
    


def compute_iou_yolo(output_yolo, labels_yolo):
    '''
    计算YOLO检测头输出与真实标签的IoU

    参数：
        output_yolo: torch.Tensor, 尺寸 (batch_size, Sx, Sy, num_classes + B*5)
        labels_yolo: torch.Tensor, 尺寸 (batch_size, Sx, Sy, num_classes + B*5)
    
    返回：
        mean_iou: float, 平均IoU值
        accuracy: float, 类别预测准确率
    '''
    batch_size = output_yolo.shape[0]
    num_classes = 5  # 假设5个类别，从代码其他部分推断
    
    # 提取类别预测和目标
    pred_class = output_yolo[..., :num_classes]
    target_class = labels_yolo[..., :num_classes]
    
    # 提取边界框预测(cx, cy, w, h, conf)
    pred_boxes = output_yolo[..., num_classes:num_classes+5]
    target_boxes = labels_yolo[..., num_classes:num_classes+5]
    
    # 计算类别准确率 - 只考虑有目标的网格
    target_obj_mask = (target_boxes[..., 4] > 0.5)  # 目标存在的掩码
    if target_obj_mask.sum() > 0:
        # 对于每个预测类别，找到概率最高的类别
        pred_class_idx = torch.argmax(pred_class, dim=-1)
        target_class_idx = torch.argmax(target_class, dim=-1)
        
        # 只计算有目标的网格的准确率
        correct = (pred_class_idx[target_obj_mask] == target_class_idx[target_obj_mask]).float()
        accuracy = correct.sum() / (target_obj_mask.sum() + 1e-6)
    else:
        accuracy = torch.tensor(0.0, device=output_yolo.device)
    
    # 计算IoU - 只考虑有目标的网格
    ious = torch.zeros_like(target_boxes[..., 0])  # 初始化IoU矩阵
    
    # 将边界框格式从[cx, cy, w, h]转换为[x1, y1, x2, y2]以计算IoU
    pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2
    
    target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
    target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
    target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
    target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2
    
    # 计算交集区域
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    
    # 计算交集面积，确保宽高为正
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    intersection = inter_w * inter_h
    
    # 计算各自面积
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    
    # 计算并集
    union = pred_area + target_area - intersection + 1e-6
    
    # 计算IoU
    ious = intersection / union
    
    # 只考虑有目标的网格的IoU
    if target_obj_mask.sum() > 0:
        mean_iou = ious[target_obj_mask].mean()
    else:
        mean_iou = torch.tensor(0.0, device=output_yolo.device)
    
    return mean_iou.item(), accuracy.item()
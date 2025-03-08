
import torch
import numpy as np
import cv2
import torchvision.ops as ops
import math
from torch.nn import functional as F

# from dataloader import Comp0249Dataset

def draw_the_box(image, boxes, nms_threshold=0.45):
    """在图像上绘制边界框，使用NMS减少重复框"""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # 转换为OpenCV可处理的格式
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # 确保图像是3通道的
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # 获取图像尺寸
    h, w = image.shape[:2]
    
    # 打印边界框形状和示例值，帮助调试
    print(f"边界框形状: {boxes.shape}")
    if boxes.size > 0 and len(boxes.shape) >= 3:
        print(f"网格[0,0]内容: {boxes[0, 0, :]}") 
    
    try:
        S_h, S_w = boxes.shape[:2]
        detection_count = 0
        confidence_threshold = 0.3
        
        # 收集所有有效框的数据用于NMS
        all_boxes = []  # 格式: [x1, y1, x2, y2]
        all_scores = []  # 置信度
        all_classes = []  # 类别
        all_metadata = []  # 存储类别概率等其他信息
        
        # 计算每个网格单元的尺寸
        cell_w = w / S_w
        cell_h = h / S_h

        # 第一步：收集所有可能的框
        for i in range(S_h):
            for j in range(S_w):
                class_probs = boxes[i, j, :5]  # 前5个是类别概率
                x = boxes[i, j, 5]  # x坐标在第6个位置
                y = boxes[i, j, 6]  # y坐标在第7个位置
                w_box = boxes[i, j, 7]  # 宽度在第8个位置
                h_box = boxes[i, j, 8]  # 高度在第9个位置
                confidence = boxes[i, j, 9]  # 置信度在第10个位置
                
                if confidence > confidence_threshold:
                    # 坐标转换
                    cx = (x + j) * cell_w
                    cy = (y + i) * cell_h
                    box_w = w_box * w
                    box_h = h_box * h
                    
                    # 计算左上和右下坐标
                    x1 = max(0, int(cx - box_w/2))
                    y1 = max(0, int(cy - box_h/2))
                    x2 = min(w-1, int(cx + box_w/2))
                    y2 = min(h-1, int(cy + box_h/2))
                    
                    # 确定类别
                    class_idx = np.argmax(class_probs) + 1  # 因为类别从1开始
                    class_prob = class_probs[class_idx-1]
                    
                    # 保存框信息
                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(confidence)
                    all_classes.append(class_idx)
                    all_metadata.append({
                        'class_prob': class_prob,
                        'class_idx': class_idx
                    })
        
        print(f"检测到 {len(all_boxes)} 个初始框")
        
        # 应用增强版NMS
        if len(all_boxes) > 0:
            keep_indices, merged_boxes, merged_scores, merged_classes = enhanced_nms(
                all_boxes, all_scores, all_classes, 
                iou_threshold=0.2,  # 传统NMS阈值
                merge_threshold=0.01  # 框融合阈值
            )
            
            print(f"融合后剩余 {len(merged_boxes)} 个框")
            
            # 绘制融合后的框
            for i in range(len(merged_boxes)):
                detection_count += 1
                x1, y1, x2, y2 = [int(c) for c in merged_boxes[i]]
                class_idx = merged_classes[i]
                confidence = merged_scores[i]
                
                # 根据类别选择颜色
                color = (0, 255, 0)  # 默认绿色
                if class_idx == 1:
                    color = (255, 0, 0)  # Car - 蓝色
                elif class_idx == 2:
                    color = (0, 0, 255)  # Pedestrian - 红色
                elif class_idx == 3:
                    color = (255, 255, 0)  # Bicyclist - 青色
                elif class_idx == 4:
                    color = (255, 0, 255)  # MotorcycleScooter - 紫色
                elif class_idx == 5:
                    color = (0, 255, 255)  # Truck_Bus - 黄色
                
                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # 添加类别标签
                class_prob = 0.0
                if len(keep_indices) > 0:
                    # 如果是融合框，取最高的类别概率
                    orig_indices = keep_indices[i] if isinstance(keep_indices[i], list) else [keep_indices[i]]
                    probs = [all_metadata[idx]['class_prob'] for idx in orig_indices if idx < len(all_metadata)]
                    class_prob = max(probs) if probs else 0.0
                
                # 在框上标记是否为融合框
                is_merged = "merged" if isinstance(keep_indices[i], list) and len(keep_indices[i]) > 1 else ""
                label = f"{is_merged} C{class_idx}: {class_prob:.2f} Conf:{confidence:.2f}"
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        print(f"绘制了 {detection_count} 个边界框")
    
    except Exception as e:
        print(f"绘制边界框时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return image

def enhanced_nms(boxes, scores, classes, iou_threshold=0.45, merge_threshold=0.25, distance_threshold=50):
    """
    增强版NMS：应用标准NMS后，对仍然重叠的框进行融合
    
    参数:
        boxes: 边界框坐标 [x1, y1, x2, y2] 的列表
        scores: 每个边界框的置信度
        classes: 每个边界框的类别
        iou_threshold: 标准NMS的IoU阈值
        merge_threshold: 框融合的IoU阈值
    
    返回:
        保留的边界框索引列表, 融合后的边界框列表, 融合后的置信度, 融合后的类别
    """

    
    if len(boxes) == 0:
        return [], [], [], []
    
    # 转换为numpy数组
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    
    # 按类别分组进行NMS
    keep_indices = []
    kept_indices_map = {}  # 为了正确追踪原始索引
    unique_classes = np.unique(classes)
    
    # 标准NMS处理
    for cls in unique_classes:
        cls_indices = np.where(classes == cls)[0]
        if len(cls_indices) == 0:
            continue
            
        cls_boxes = boxes[cls_indices]
        cls_scores = scores[cls_indices]
        
        # 使用torchvision的nms函数
        cls_boxes_tensor = torch.tensor(cls_boxes, dtype=torch.float32)
        cls_scores_tensor = torch.tensor(cls_scores, dtype=torch.float32)
        keep_tensor = ops.nms(cls_boxes_tensor, cls_scores_tensor, iou_threshold)
        
        keep_cls = [cls_indices[i] for i in keep_tensor.cpu().numpy()]
        for i, idx in enumerate(keep_cls):
            kept_indices_map[len(keep_indices) + i] = idx  # 记录索引映射
        keep_indices.extend(keep_cls)

        print(f"NMS后保留了{len(keep_indices)}个框")
    
    # 没有框或只有一个框时直接返回
    if len(keep_indices) <= 1:
        return keep_indices, boxes[keep_indices] if keep_indices else [], scores[keep_indices] if keep_indices else [], classes[keep_indices] if keep_indices else []
    
    # 处理NMS后的框，查找仍有重叠的框
    kept_boxes = boxes[keep_indices]
    kept_scores = scores[keep_indices]
    kept_classes = classes[keep_indices]
    
    # 计算IoU矩阵
    def calculate_iou(box1, box2):
        """
        计算两个矩形框的 IoU（Intersection over Union），
        自动处理坐标顺序问题
        
        :param box1: (x1, y1, x2, y2)
        :param box2: (x1, y1, x2, y2)
        :return: IoU 值（0~1）
        """
        # 解析并修正 box1 的坐标，确保 x1<x2, y1<y2
        x1_1, y1_1, x2_1, y2_1 = box1
        if x1_1 > x2_1:
            x1_1, x2_1 = x2_1, x1_1
        if y1_1 > y2_1:
            y1_1, y2_1 = y2_1, y1_1
        
        # 解析并修正 box2 的坐标，确保 x1<x2, y1<y2
        x1_2, y1_2, x2_2, y2_2 = box2
        if x1_2 > x2_2:
            x1_2, x2_2 = x2_2, x1_2
        if y1_2 > y2_2:
            y1_2, y2_2 = y2_2, y1_2
        
        # 打印修正后的坐标，便于调试
        print(f"修正后的框1: [{x1_1}, {y1_1}, {x2_1}, {y2_1}]")
        print(f"修正后的框2: [{x1_2}, {y1_2}, {x2_2}, {y2_2}]")
    
        # 计算交集坐标（交集的左上角 & 右下角）
        ix1 = max(x1_1, x1_2)
        iy1 = max(y1_1, y1_2)
        ix2 = min(x2_1, x2_2)
        iy2 = min(y2_1, y2_2)
    
        # 计算交集区域的宽度和高度
        iw = max(0, ix2 - ix1)  # 防止负数
        ih = max(0, iy2 - iy1)
    
        # 交集面积
        intersection_area = iw * ih
    
        # 计算 box1 和 box2 的面积
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
        # 计算并集面积（Union = A + B - Intersection）
        union_area = area1 + area2 - intersection_area
    
        # 计算 IoU，防止除以 0
        iou = intersection_area / union_area if union_area > 0 else 0
        
        # 打印计算结果，便于调试
        print(f"交集面积: {intersection_area}, 并集面积: {union_area}, IoU: {iou:.4f}")
    
        return iou
    
    # 创建标记数组，标记已经被融合的框
    is_merged = np.zeros(len(kept_boxes), dtype=bool)
    merged_boxes = []
    merged_scores = []
    merged_classes = []
    merged_indices = []  # 跟踪每个合并框对应的原始索引
    merge_count = 0

      # 框融合
    for i in range(len(kept_boxes)):
        if is_merged[i]:
            continue
            
        # 找到与当前框重叠的所有框
        current_box = kept_boxes[i]
        current_score = kept_scores[i]
        current_class = kept_classes[i]
        
        # 当前框的中心点
        cx1 = (current_box[0] + current_box[2]) / 2
        cy1 = (current_box[1] + current_box[3]) / 2

        overlaps = []
        for j in range(len(kept_boxes)):
            if i != j and not is_merged[j] and kept_classes[j] == current_class:
                # 计算IoU
                iou = calculate_iou(current_box, kept_boxes[j])
                
                # 计算两个框的中心点距离
                other_box = kept_boxes[j]
                cx2 = (other_box[0] + other_box[2]) / 2
                cy2 = (other_box[1] + other_box[3]) / 2
                distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                
                # 打印所有调试信息 
                print(f"框{i}与框{j}的IoU={iou:.5f}, 距离={distance:.1f}px")
                
                # 两种条件之一满足就融合：IoU足够大或距离足够近
                if iou > merge_threshold or distance < distance_threshold:
                    overlaps.append(j)
                    print(f"将融合框{j} - IoU={iou:.5f}, 距离={distance:.1f}px")
        
        # 如果有重叠框，融合它们
        if overlaps:
            merge_count += 1
            print(f"发现融合机会: 框{i}与{len(overlaps)}个框({overlaps})重叠")
            # 标记所有要融合的框
            is_merged[i] = True
            for j in overlaps:
                is_merged[j] = True
                
            # 收集所有要融合的框和对应的原始索引
            indices_to_merge = [keep_indices[i]] + [keep_indices[j] for j in overlaps]
            merged_indices.append(indices_to_merge)
            
            # 融合框逻辑保持不变...
            boxes_to_merge = [current_box] + [kept_boxes[j] for j in overlaps]
            scores_to_merge = [current_score] + [kept_scores[j] for j in overlaps]
            
            # 融合框：取最小x1,y1和最大x2,y2
            x1 = min(box[0] for box in boxes_to_merge)
            y1 = min(box[1] for box in boxes_to_merge)
            x2 = max(box[2] for box in boxes_to_merge)
            y2 = max(box[3] for box in boxes_to_merge)
            
            # 计算融合框的平均置信度
            avg_score = sum(scores_to_merge) / len(scores_to_merge)
            
            # 添加融合后的框
            merged_boxes.append([x1, y1, x2, y2])
            merged_scores.append(avg_score)
            merged_classes.append(current_class)
        elif not is_merged[i]:
            # 如果当前框没有被融合，直接添加
            merged_boxes.append(current_box)
            merged_scores.append(current_score)
            merged_classes.append(current_class)
            merged_indices.append([keep_indices[i]])  # 单个框的原始索引
            is_merged[i] = True
    
    print(f"NMS后剩余{len(keep_indices)}个框，找到{merge_count}次融合机会，融合后剩余{len(merged_boxes)}个框")
    
    # 返回新版本的索引和结果
    return merged_indices, np.array(merged_boxes), np.array(merged_scores), np.array(merged_classes)

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
    mse_loss = F.mse_loss(pred_class, target_class, reduction='none')
    pt = torch.exp(-mse_loss)
    
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
    focal_loss = focal_weight * mse_loss * obj_expanded
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
        pred_wh * obj_expanded_wh, 
        target_wh * obj_expanded_wh, 
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


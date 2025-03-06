import cv2 
import numpy as np
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


import torch
import numpy as np
import cv2

def segmentation_to_yolo(label, S=7, num_classes=20, B=2, scale=1):
    """
    将分割标签 (H, W) 转换为 YOLO 格式标签 (S, S, num_classes + B*5)
    参数：
        label: torch.Tensor, 分割标签，尺寸为 (H, W)，像素值 0 表示背景，其余值对应类别编号（建议连续）
        S: YOLO 网格尺寸（例如 7 表示 7x7）
        num_classes: 类别数
        B: 每个网格预测的边框数，通常这里只用第一个框进行标注，其余保持0
    返回：
        yolo_label: torch.Tensor, 尺寸 (S, S, num_classes + B*5)
                    格式为：前 num_classes 维：目标类别；
                            后面 B*5 维，每个边框组成 [cx, cy, w, h, confidence]
    """
    # 将 label 转为 numpy 数组
    label_np = label.cpu().numpy().astype(np.uint8)
    H, W = label_np.shape
    yolo_label = np.zeros((S, S, B * 5 + num_classes), dtype=np.float32)

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


        # 确定目标中心所在的网格单元[0-(S-1)]
        grid_x = int(norm_cx * S)
        grid_y = int(norm_cy * S)
        grid_x = min(grid_x, S - 1)
        grid_y = min(grid_y, S - 1)

        # 设置该网格单元的类别信息与边框信息（这里只填充第一个边框，后续边框保持0）
        yolo_label[grid_y, grid_x, class_id - 1] = 1
        # 计算相对于网格的偏移量
        rel_cx = norm_cx * S - grid_x  # x偏移，范围[0,1]
        rel_cy = norm_cy * S - grid_y  # y偏移，范围[0,1]

        # 存储相对坐标
        yolo_label[grid_y, grid_x, num_classes:num_classes+5] = np.array([rel_cx, rel_cy, norm_w, norm_h, 1.0], dtype=np.float32)

    # 转换回 torch.Tensor
    return torch.from_numpy(yolo_label)



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
def segmentation_to_yolov3(label, w, h, num_classes=20, B=2, scale=1):
    """
    将分割标签转换为YOLOv3格式的三个不同尺度的标签
    参数：
        label: 分割标签
        w: 图像宽度
        h: 图像高度
        num_classes: 类别数
        B: 每个网格预测的边框数
        scale: 面积过滤的比例因子
    返回：
        三个不同尺度的YOLO格式标签：大、中、小 (从小到大排序)
    """
    return (
        segmentation_to_yolov3_1(label, math.ceil(w/32), math.ceil(h/32), num_classes, B, scale),
        segmentation_to_yolov3_1(label, w//16, h//16, num_classes, B, scale),
        segmentation_to_yolov3_1(label, w//8, h//8, num_classes, B, scale)
    )











def check_binary(matrix: torch.Tensor) -> bool:
    """
    检查输入的矩阵是否只包含 0 和 1

    参数：
        matrix (torch.Tensor): 输入张量

    返回：
        bool: 如果只包含0和1返回True，否则返回False
    """
    return torch.all((matrix == 0) | (matrix == 1)).item()

def cx_cy_to_corners(cx, cy, w, h):
    '''
    input: cx, cy, w, h - the center x, center y, width, and height of the box
    output: x1, y1, x2, y2 - the coordinates of the top left and bottom right corners of the box
    '''
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return x1, y1, x2, y2

def draw_the_yolo_label(image, yolo_label):
    '''
    input: image - the image to draw the box on
                    yolo_label: torch.Tensor, 尺寸 (S, S, num_classes + B*5)
                    格式为：前 num_classes 维：one-hot 目标类别；
                            后面 B*5 维，每个边框组成 [cx, cy, w, h, confidence]
    output: image - the image with the box drawn on it (H, W)
    '''
    H, W, C = image.shape
    class_yolo = yolo_label[..., :5]
    bbox_yolo = yolo_label[..., 5:]
    for i in range(5):
        # 得到当前类别的掩码，类型为 bool
        mask = (class_yolo[..., i] > 0)
        # 如果该类别没有像素，则跳过
        if np.count_nonzero(mask) == 0:
            continue
        Sy, Sx = mask.shape
        # 获得该类别像素的行和列索引
        rows, cols = np.where(mask)
        class_position = np.hstack((rows.reshape(-1, 1), cols.reshape(-1, 1)))

        for _ in range(len(class_position)):
            y_index, x_index = class_position[_] # 23, 30

            selected_yolo = bbox_yolo[y_index, x_index, :]

            position = selected_yolo[..., :4].numpy().flatten()

            position[0] = (x_index + position[0]) / Sx
            position[1] = (y_index + position[1]) / Sy
            position[0], position[1], position[2], position[3] = cx_cy_to_corners(*position)

            position = position * np.array([W, H, W, H])
            position = position.astype(np.int32)
            # 保证 image 内存连续
            image = np.ascontiguousarray(image)
            # 绘制矩形框
            image = cv2.rectangle(image, (position[0], position[1]), (position[2], position[3]), (255, 25*i, 255), 2)
        
    return image



def compute_iou_yolov3(pred, target, w, h, num_classes=20, B=2):
    '''
    计算YOLOv3预测结果与目标标签之间的IoU值
    
    参数：
        pred: 模型预测结果，形状为 (Sy, Sx, num_classes + B*5)
        target: 目标标签，形状为 (Sy, Sx, num_classes + B*5)
        w, h: 原始图像的宽度和高度
        num_classes: 类别数量
        B: 每个网格预测的边界框数量
    
    返回：
        class_iou: 各类别IoU的字典
        mean_iou: 所有类别的平均IoU值
    '''
    # 提取类别预测和目标
    pred_class = pred[..., :num_classes]
    target_class = target[..., :num_classes]
    
    # 提取边界框信息
    pred_boxes = pred[..., num_classes:].reshape(-1, B, 5)  # (Sy*Sx, B, 5)
    target_boxes = target[..., num_classes:].reshape(-1, B, 5)  # (Sy*Sx, B, 5)
    
    Sy, Sx = pred_class.shape[:2]  # 获取网格尺寸
    
    # 存储每个类别的IoU值
    class_iou = {}
    total_iou = 0
    valid_classes = 0
    
    # 遍历所有类别
    for c in range(num_classes):
        # 找到包含该类别的网格
        pred_class_mask = (pred_class[..., c] > 0.5).flatten()  # (Sy*Sx)
        target_class_mask = (target_class[..., c] > 0.5).flatten()  # (Sy*Sx)
        
        if not target_class_mask.any():  # 如果目标中没有该类别，跳过
            continue
        
        # 获取该类别的预测和目标边界框
        class_pred_boxes = pred_boxes[pred_class_mask]  # (N, B, 5)，N为预测该类的网格数
        class_target_boxes = target_boxes[target_class_mask]  # (M, B, 5)，M为目标中该类的网格数
        
        if len(class_pred_boxes) == 0 or len(class_target_boxes) == 0:
            class_iou[c] = 0
            continue
        
        # 只考虑置信度最高的边界框
        class_pred_boxes = class_pred_boxes[:, torch.argmax(class_pred_boxes[..., 4], dim=1)]  # (N, 5)
        class_target_boxes = class_target_boxes[:, torch.argmax(class_target_boxes[..., 4], dim=1)]  # (M, 5)
        
        # 预测框转换为绝对坐标 (x1, y1, x2, y2)
        pred_absolute_boxes = []
        for i, box in enumerate(class_pred_boxes):
            # 获取对应网格索引
            grid_idx = torch.where(pred_class_mask)[0][i]
            gy, gx = grid_idx // Sx, grid_idx % Sx  # 将一维索引转为二维网格坐标
            
            # 转换为图像中的中心坐标
            cx = (gx + box[0]) * w / Sx  # 转换为图像绝对坐标
            cy = (gy + box[1]) * h / Sy
            box_w = box[2] * w  # 宽高是相对整个图像的比例
            box_h = box[3] * h
            
            # 转换为左上右下角坐标 - 使用torch函数替代max/min
            x1 = torch.clamp(cx - box_w / 2, min=0)
            y1 = torch.clamp(cy - box_h / 2, min=0)
            x2 = torch.clamp(cx + box_w / 2, max=w)
            y2 = torch.clamp(cy + box_h / 2, max=h)
            
            # 使用stack创建新张量，保留原始设备
            pred_absolute_boxes.append(torch.stack([x1, y1, x2, y2]))

        # 目标框转换为绝对坐标 - 同样应用修改
        target_absolute_boxes = []
        for i, box in enumerate(class_target_boxes):
            # 获取对应网格索引
            grid_idx = torch.where(target_class_mask)[0][i]
            gy, gx = grid_idx // Sx, grid_idx % Sx
            
            # 转换为图像中的中心坐标
            cx = (gx + box[0]) * w / Sx
            cy = (gy + box[1]) * h / Sy
            box_w = box[2] * w
            box_h = box[3] * h
            
            # 转换为左上右下角坐标 - 使用torch函数替代max/min
            x1 = torch.clamp(cx - box_w / 2, min=0)
            y1 = torch.clamp(cy - box_h / 2, min=0)
            x2 = torch.clamp(cx + box_w / 2, max=w)
            y2 = torch.clamp(cy + box_h / 2, max=h)
            
            # 使用stack创建新张量，保留原始设备
            target_absolute_boxes.append(torch.stack([x1, y1, x2, y2]))
        
        # 如果转换后没有有效框，跳过
        if not pred_absolute_boxes or not target_absolute_boxes:
            class_iou[c] = 0
            continue
        
        # 计算每个预测框与所有目标框的IoU
        pred_boxes_tensor = torch.stack(pred_absolute_boxes)
        target_boxes_tensor = torch.stack(target_absolute_boxes)
        
        # 计算IoU矩阵
        ious = []
        for pred_box in pred_boxes_tensor:
            box_ious = []
            for target_box in target_boxes_tensor:
                # 计算交集区域 - 使用torch函数替代max/min
                x1 = torch.maximum(pred_box[0], target_box[0])
                y1 = torch.maximum(pred_box[1], target_box[1])
                x2 = torch.minimum(pred_box[2], target_box[2])
                y2 = torch.minimum(pred_box[3], target_box[3])
                
                # 计算交集面积
                w_inter = torch.clamp(x2 - x1, min=0)
                h_inter = torch.clamp(y2 - y1, min=0)
                intersection = w_inter * h_inter
                
                # 计算并集面积
                pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                target_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
                union = pred_area + target_area - intersection
                
                # 计算IoU
                iou = intersection / (union + 1e-6)  # 防止除零
                box_ious.append(iou.item())  # 转换为Python标量
            
            if box_ious:  # 确保列表非空
                ious.append(max(box_ious))  # 取最大IoU
        
        # 计算该类别的平均IoU
        class_iou[c] = sum(ious) / len(ious) if ious else 0
        total_iou += class_iou[c]
        valid_classes += 1
    
    # 计算所有类别的平均IoU
    mean_iou = total_iou / valid_classes if valid_classes > 0 else 0
    
    return class_iou, mean_iou


from torch.nn import functional as F
# def yolo_loss(predictions, targets, Sx, Sy, B=1, C=5, lambda_coord=5, lambda_noobj=0.5):
#     """
#     YOLO 损失函数计算（支持 batch 维度）
#     """
#     batch = predictions.shape[0]  # 获取 batch 大小

#     # predictions[..., :C] = torch.sigmoid(predictions[..., :C])
#     # predictions[..., C+4:C+4+B*5:5] = torch.sigmoid(predictions[..., C+4:4+C+B*5:5])

#     # 类别损失
#     class_loss = F.mse_loss(predictions[..., :C], targets[..., :C], reduction='sum')

#     # 计算坐标损失
#     coord_loss = lambda_coord * (
#         F.mse_loss(predictions[..., C:C+2], targets[..., C:C+2], reduction='sum') +
#         F.mse_loss(torch.sqrt(torch.abs(predictions[..., C+2:C+4] + 1e-6)), torch.sqrt(torch.abs(targets[..., C+2:C+4] + 1e-6)), reduction='sum')
#     )

#     # 置信度损失
#     obj_loss = F.mse_loss(predictions[..., C+4], targets[..., C+4], reduction='sum')
#     noobj_loss = lambda_noobj * F.mse_loss(predictions[..., C+4] * (1 - targets[..., C+4]), targets[..., C+4] * (1 - targets[..., C+4]), reduction='sum')

#     # 总损失
#     total_loss = class_loss + coord_loss + obj_loss + noobj_loss
#     return total_loss/4

def yolo_loss(predictions, targets, Sx, Sy, B=1, C=5, lambda_coord=5, lambda_noobj=0.5, gamma=2.0, alpha=0.25):
    """修正后的YOLO损失函数计算"""
    batch = predictions.shape[0]  # 获取batch大小
    
    # 获取预测和目标
    pred_class = predictions[..., :C]
    target_class = targets[..., :C]
    
    # 创建有目标的掩码
    obj_mask = (targets[..., C+4] > 0.5)
    noobj_mask = ~obj_mask
    
    # 1. 只对有目标的网格计算类别损失(使用Focal Loss)
    bce_loss = F.binary_cross_entropy_with_logits(pred_class, target_class, reduction='none')
    pt = torch.exp(-bce_loss)
    
    # 区分正负样本的alpha
    pos_mask = (target_class > 0.5)
    neg_mask = ~pos_mask
    alpha_weight = pos_mask * alpha + neg_mask * (1-alpha)
    
    # 应用Focal Loss权重
    focal_weight = alpha_weight * (1 - pt) ** gamma
    
    # 只对有目标的网格计算类别损失
    obj_expanded = obj_mask.unsqueeze(-1).expand_as(pred_class)
    focal_loss = focal_weight * bce_loss * obj_expanded
    class_loss = focal_loss.sum()
    
    # 2. 只对有目标的网格计算坐标损失
    # 提取坐标预测
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
        reduction='sum'
    )
    
    # 宽高损失使用平方根变换
    wh_loss = F.mse_loss(
        torch.sqrt(torch.abs(pred_wh) + 1e-6) * obj_expanded_wh, 
        torch.sqrt(torch.abs(target_wh) + 1e-6) * obj_expanded_wh, 
        reduction='sum'
    )
    
    coord_loss = lambda_coord * (xy_loss + wh_loss)
    
    # 3. 置信度损失计算不变
    obj_conf_loss = F.mse_loss(
        predictions[..., C+4][obj_mask], 
        targets[..., C+4][obj_mask], 
        reduction='sum'
    )
    
    noobj_conf_loss = lambda_noobj * F.mse_loss(
        predictions[..., C+4][noobj_mask], 
        targets[..., C+4][noobj_mask], 
        reduction='sum'
    )
    
    # 4. 更合理地归一化损失
    # 计算有目标的网格数量
    num_obj = obj_mask.sum().float() + 1e-6
    
    # 归一化损失
    total_loss = class_loss + coord_loss + obj_conf_loss + noobj_conf_loss
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
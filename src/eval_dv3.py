import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from collections import defaultdict

# 导入自定义模块
from train_deeplabv3 import compute_iou, FocalLoss, calculate_class_weights
from model import ResNetHead, ASPP, DeepLabV3PlusDecoder, ConvBlock, StageBlock1, StageBlock2, StageBlock3, StageBlock4, YOLOHead, TotalDeepLabV3Plus, StageBlockmid, StageBlock4_2
from dataloader import Comp0249Dataset
from utils import compute_iou_yolo
import torchvision.ops as ops

def enhanced_nms(boxes, scores, classes, iou_threshold=0.45, merge_threshold=0.25):
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
        keep_indices.extend(keep_cls)
    
    # 没有框或只有一个框时直接返回
    if len(keep_indices) <= 1:
        return keep_indices, boxes[keep_indices] if keep_indices else [], scores[keep_indices] if keep_indices else [], classes[keep_indices] if keep_indices else []
    
    # 处理NMS后的框，查找仍有重叠的框
    kept_boxes = boxes[keep_indices]
    kept_scores = scores[keep_indices]
    kept_classes = classes[keep_indices]
    
    # 计算IoU矩阵
    def calculate_iou(box1, box2):
        """计算两个框的IoU"""
        # 获取交集
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    # 创建标记数组，标记已经被融合的框
    is_merged = np.zeros(len(kept_boxes), dtype=bool)
    merged_boxes = []
    merged_scores = []
    merged_classes = []
    
    # 框融合
    for i in range(len(kept_boxes)):
        if is_merged[i]:
            continue
            
        # 找到与当前框重叠的所有框
        current_box = kept_boxes[i]
        current_score = kept_scores[i]
        current_class = kept_classes[i]
        
        overlaps = []
        for j in range(len(kept_boxes)):
            if i != j and not is_merged[j] and kept_classes[j] == current_class:
                iou = calculate_iou(current_box, kept_boxes[j])
                if iou > merge_threshold:
                    overlaps.append(j)
        
        # 如果有重叠框，融合它们
        if overlaps:
            # 标记所有要融合的框
            is_merged[i] = True
            for j in overlaps:
                is_merged[j] = True
                
            # 收集所有要融合的框
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
            is_merged[i] = True
    
    print(f"NMS后剩余 {len(keep_indices)} 个框，融合后剩余 {len(merged_boxes)} 个框")
    
    # 返回原始索引和融合后的结果
    return keep_indices, np.array(merged_boxes), np.array(merged_scores), np.array(merged_classes)
# 修改draw_the_box函数以使用NMS

# 创建保存结果的目录
os.makedirs('results/evaluation', exist_ok=True)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
print("正在加载模型...")
model_path = 'results/deeplabmodelfullfinal.pth'
model = torch.load(model_path, weights_only=False)
model.eval()
model.to(device)

# 加载测试数据
print("正在加载测试数据...")
batch_size = 4  # 评估时可以用稍大的batch size
test_dataset = Comp0249Dataset('data/CamVid', "val", scale=1, transform=None, target_transform=None)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# 创建类别映射
class_names = {
    0: "Background",
    1: "Car",
    2: "Pedestrian",
    3: "Bicyclist",
    4: "MotorcycleScooter",
    5: "Truck_Bus"
}

# 初始化评估指标存储
segmentation_metrics = {
    'total_loss': [],
    'mean_iou': [],
    'class_iou': defaultdict(list),
    'confusion_matrix': np.zeros((6, 6), dtype=np.int64)  # 假设6个类别
}

detection_metrics = {
    'yolo_iou': [],
    'class_accuracy': defaultdict(list),
    'detection_count': defaultdict(int)
}

# 加载FocalLoss以计算验证损失
class_weights = calculate_class_weights(test_dataset)
class_weights = class_weights.to(device)
criterion = FocalLoss(alpha=class_weights, gamma=2.0)

criterion = nn.CrossEntropyLoss()

# 主评估循环
print("开始评估...")
all_images = []
all_true_masks = []
all_pred_masks = []
all_pred_boxes = []
all_true_boxes = []

with torch.no_grad():
    for idx, (images, labels) in enumerate(tqdm(test_loader, desc="评估")):
        images = images.to(device, dtype=torch.float32)
        labels_segment = labels[0].to(device, dtype=torch.long)
        labels_yolo = labels[1].to(device, dtype=torch.float)
        
        # 模型推理
        pred_segment, pred_yolo = model(images)
        # 创建伪logits以匹配预期格式
        # b, h, w = labels_segment.size()
        # num_classes = 6
        # pred_segment = torch.zeros(b, num_classes, h, w, device=device)
        # # 为每个标签在对应位置设置高值(10.0)，模拟softmax前的logits
        # for i in range(num_classes):
        #     pred_segment[:, i, :, :] = (labels_segment == i).float() * 10.0

        # # 保持YOLO预测与标签相同
        # pred_yolo = labels_yolo

        
        # 计算分割损失和IoU
        seg_loss = criterion(pred_segment, labels_segment)
        iou_dict, batch_mean_iou = compute_iou(pred_segment, labels_segment, num_classes=6)
        
        # 保存分割指标
        segmentation_metrics['total_loss'].append(seg_loss.item())
        segmentation_metrics['mean_iou'].append(batch_mean_iou)
        
        # 为每个类别保存IoU
        for cls in range(6):
            class_key = f'class_{cls}'
            if class_key in iou_dict:
                iou_value = iou_dict[class_key].item() if isinstance(iou_dict[class_key], torch.Tensor) else iou_dict[class_key]
                if not np.isnan(iou_value):
                    segmentation_metrics['class_iou'][cls].append(iou_value)
        
        # 计算预测标签用于混淆矩阵
        pred_labels = torch.argmax(pred_segment, dim=1)
        
        # 修改目标检测评估部分
        
        # 更新混淆矩阵
        for b in range(labels_segment.size(0)):
            true_flat = labels_segment[b].cpu().numpy().flatten()
            pred_flat = pred_labels[b].cpu().numpy().flatten()
            cm = confusion_matrix(true_flat, pred_flat, labels=range(6))
            segmentation_metrics['confusion_matrix'] += cm
        
        # 计算目标检测指标
        try:
            yolo_iou, class_acc = compute_iou_yolo(pred_yolo, labels_yolo)
            detection_metrics['yolo_iou'].append(yolo_iou)
            
            # 如果class_acc是浮点数，则将其存储为总体准确率
            if isinstance(class_acc, float):
                detection_metrics['class_accuracy']['overall'].append(class_acc)
            # 如果class_acc是字典，则存储每个类别的准确率
            elif isinstance(class_acc, dict):
                for cls_name, acc_value in class_acc.items():
                    detection_metrics['class_accuracy'][cls_name].append(acc_value)
                    # 统计每个类别的检测数量
                    cls_idx = int(cls_name.split('_')[1]) if '_' in cls_name else 0
                    if acc_value > 0.5:  # 假设准确率大于0.5算检测成功
                        detection_metrics['detection_count'][cls_idx] += 1
        except Exception as e:
            print(f"计算目标检测指标出错: {str(e)}")
            print(f"错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
                
        # 保存一些图像用于可视化(最多保存10个batch)
        if idx < 10:
            for b in range(min(images.size(0), 2)):  # 每个batch最多保存2张图
                img = images[b].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                true_mask = labels_segment[b].cpu().numpy()
                pred_mask = pred_labels[b].cpu().numpy()
                
                all_images.append(img)
                all_true_masks.append(true_mask)
                all_pred_masks.append(pred_mask)
                
                # 保存目标检测结果
                all_true_boxes.append(labels_yolo[b].cpu().numpy())
                all_pred_boxes.append(pred_yolo[b].cpu().numpy())
# 计算平均指标
avg_loss = np.mean(segmentation_metrics['total_loss'])
avg_miou = np.mean(segmentation_metrics['mean_iou'])
avg_yolo_iou = np.mean(detection_metrics['yolo_iou'])

# 计算每个类别的平均IoU
class_iou_avg = {}
for cls in range(6):
    if segmentation_metrics['class_iou'][cls]:
        class_iou_avg[cls] = np.mean(segmentation_metrics['class_iou'][cls])
    else:
        class_iou_avg[cls] = float('nan')

# 计算每个类别的检测准确率
class_det_acc = {}
for cls in range(1, 6):  # 跳过背景类
    if detection_metrics['class_accuracy'][cls]:
        class_det_acc[cls] = np.mean(detection_metrics['class_accuracy'][cls])
    else:
        class_det_acc[cls] = float('nan')

# 打印评估结果
print("\n=== 分割评估结果 ===")
print(f"平均损失: {avg_loss:.4f}")
print(f"平均mIoU: {avg_miou:.4f}")
print("各类别IoU:")
for cls in range(6):
    class_name = class_names[cls]
    iou_val = class_iou_avg[cls] if cls in class_iou_avg else float('nan')
    print(f"  - {class_name}: {iou_val:.4f}")

print("\n=== 目标检测评估结果 ===")
print(f"平均YOLO IoU: {avg_yolo_iou:.4f}")
print("各类别检测准确率:")
for cls in range(1, 6):  # 跳过背景类
    class_name = class_names[cls]
    acc_val = class_det_acc[cls] if cls in class_det_acc else float('nan')
    det_count = detection_metrics['detection_count'][cls]
    print(f"  - {class_name}: {acc_val:.4f} (检测到{det_count}个)")

# 保存详细评估结果到CSV
results_df = pd.DataFrame({
    'Class': [class_names[i] for i in range(6)],
    'Segmentation_IoU': [class_iou_avg.get(i, float('nan')) for i in range(6)],
    'Detection_Accuracy': [class_det_acc.get(i, float('nan')) for i in range(1, 6)] + [float('nan')]
})
results_df.to_csv('results/evaluation/detailed_results.csv', index=False)

# 绘制并保存混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(segmentation_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
           xticklabels=[class_names[i] for i in range(6)],
           yticklabels=[class_names[i] for i in range(6)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('results/evaluation/confusion_matrix.png')

# 可视化一些预测结果
num_samples = min(8, len(all_images))
plt.figure(figsize=(15, 5*num_samples))

for i in range(num_samples):
    # 原始图像
    plt.subplot(num_samples, 3, i*3+1)
    plt.imshow(all_images[i])
    plt.title(f"Sample {i+1} - Input")
    plt.axis('off')
    
    # 真实分割掩码
    plt.subplot(num_samples, 3, i*3+2)
    plt.imshow(all_true_masks[i], cmap='tab20')
    plt.title(f"Sample {i+1} - Ground Truth")
    plt.axis('off')
    
    # 预测分割掩码
    plt.subplot(num_samples, 3, i*3+3)
    plt.imshow(all_pred_masks[i], cmap='tab20')
    plt.title(f"Sample {i+1} - Prediction")
    plt.axis('off')

plt.tight_layout()
plt.savefig('results/evaluation/segmentation_results.png')

# 绘制目标检测结果
plt.figure(figsize=(15, 5*num_samples))

for i in range(num_samples):
    # 原始图像
    plt.subplot(num_samples, 2, i*2+1)
    # 对真值框不应用NMS(或使用较高阈值)
    img_with_true_boxes = draw_the_box(all_images[i].copy(), all_true_boxes[i], nms_threshold=0.8)
    plt.imshow(img_with_true_boxes)
    plt.title(f"Sample {i+1} - Ground Truth Boxes")
    plt.axis('off')
    
    # 预测框应用NMS
    plt.subplot(num_samples, 2, i*2+2)
    img_with_pred_boxes = draw_the_box(all_images[i].copy(), all_pred_boxes[i], nms_threshold=0.05)
    plt.imshow(img_with_pred_boxes)
    plt.title(f"Sample {i+1} - Predicted Boxes (NMS)")
    plt.axis('off')

plt.tight_layout()
plt.savefig('results/evaluation/detection_results.png')

# 绘制类别性能条形图
plt.figure(figsize=(12, 6))

# 分割IoU条形图
classes = list(class_names.values())
ious = [class_iou_avg.get(i, 0) for i in range(6)]
colors = ['#3498db' if i not in [4, 5] else '#e74c3c' for i in range(6)]  # 少数类用红色

bars = plt.bar(classes, ious, color=colors)
plt.axhline(y=avg_miou, color='r', linestyle='--', label=f'Average mIoU: {avg_miou:.4f}')
plt.xlabel('Class')
plt.ylabel('IoU')
plt.title('Segmentation IoU by Class')
plt.xticks(rotation=45)
plt.ylim(0, 1.0)

# 标注数值
for bar, iou in zip(bars, ious):
    if not np.isnan(iou):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{iou:.3f}', ha='center', va='bottom')

plt.legend()
plt.tight_layout()
plt.savefig('results/evaluation/class_performance.png')

print("\n评估完成！结果已保存到results/evaluation/目录")
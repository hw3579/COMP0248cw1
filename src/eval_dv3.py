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
from train_deeplabv3 import compute_iou, TotalDeepLabV3Plus, FocalLoss, calculate_class_weights
from train_deeplabv3 import ResNetHead, ASPP, DeepLabV3PlusDecoder, ConvBlock, StageBlock1, StageBlock2, StageBlock3, StageBlock4, YOLOHead, TotalDeepLabV3Plus, StageBlockmid
from dataloader import Comp0249Dataset
from utils import compute_iou_yolo


# 修改NMS函数以修复计算问题并增加调试信息

def nms(boxes, scores, classes, iou_threshold=0.45):
    """
    执行非极大值抑制
    
    参数:
        boxes: 边界框坐标 [x1, y1, x2, y2] 的列表
        scores: 每个边界框的置信度
        classes: 每个边界框的类别
        iou_threshold: IoU阈值，高于此值的重叠框将被抑制
        
    返回:
        保留的边界框索引列表
    """
    if len(boxes) == 0:
        return []
        
    # 确保输入是numpy数组
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    
    # 按类别分组进行NMS
    keep_indices = []
    unique_classes = np.unique(classes)
    
    print(f"NMS统计：共 {len(boxes)} 个框，{len(unique_classes)} 个唯一类别")
    
    for cls in unique_classes:
        # 获取该类别的所有框
        cls_indices = np.where(classes == cls)[0]
        if len(cls_indices) == 0:
            continue
            
        cls_boxes = boxes[cls_indices]
        cls_scores = scores[cls_indices]
        
        print(f"  类别 {cls}: {len(cls_indices)} 个框")
        
        # 按置信度降序排序
        order = cls_scores.argsort()[::-1]
        keep_cls = []
        
        while order.size > 0:
            # 保留分数最高的框
            i = order[0]
            keep_cls.append(cls_indices[i])
            
            # 只剩一个框时结束
            if order.size == 1:
                break
                
            # 计算IoU
            # 注意：移除 +1 偏移，使用标准的IoU计算
            xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
            yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
            xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[order[1:], 2])
            yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)  # 移除+1
            h = np.maximum(0.0, yy2 - yy1)  # 移除+1
            inter = w * h
            
            area1 = (cls_boxes[i, 2] - cls_boxes[i, 0]) * (cls_boxes[i, 3] - cls_boxes[i, 1])  # 移除+1
            area2 = (cls_boxes[order[1:], 2] - cls_boxes[order[1:], 0]) * (cls_boxes[order[1:], 3] - cls_boxes[order[1:], 1])  # 移除+1
            
            iou = inter / (area1 + area2 - inter + 1e-10)  # 添加小量以防止除零
            
            # 打印一些IoU值样本
            if len(iou) > 0:
                print(f"    样本IoU值: {iou[:min(3, len(iou))]}")
                
            # 保留IoU小于阈值的框
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]  # +1是因为第一个已经处理了
            
        print(f"  类别 {cls}: 保留 {len(keep_cls)} 个框，移除 {len(cls_indices) - len(keep_cls)} 个框")
        keep_indices.extend(keep_cls)
        
    return keep_indices



# 修改draw_the_box函数以使用NMS

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
        
        # 第二步：应用NMS
        if len(all_boxes) > 0:
            keep_indices = nms(all_boxes, all_scores, all_classes, iou_threshold=nms_threshold)
            
            print(f"NMS后保留 {len(keep_indices)} 个框")
            
            # 第三步：绘制保留的框
            for i in keep_indices:
                detection_count += 1
                x1, y1, x2, y2 = all_boxes[i]
                class_idx = all_classes[i]
                confidence = all_scores[i]
                class_prob = all_metadata[i]['class_prob']
                
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
                label = f"C{class_idx}: {class_prob:.2f} Conf:{confidence:.2f}"
                cv2.putText(image, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        print(f"绘制了 {detection_count} 个边界框")
    
    except Exception as e:
        print(f"绘制边界框时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return image

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
    img_with_pred_boxes = draw_the_box(all_images[i].copy(), all_pred_boxes[i], nms_threshold=0.45)
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
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
from model import TotalDeepLabV3Plus
from dataloader import Comp0249Dataset
from utils import compute_iou_yolo, draw_the_box, calculate_precision_recall, compute_ap_from_matches

def initialize_metrics():
    """初始化评估指标字典"""
    segmentation_metrics = {
        'total_loss': [],
        'mean_iou': [],
        'class_iou': defaultdict(list),
        'confusion_matrix': np.zeros((6, 6), dtype=np.int64)
    }

    detection_metrics = {
        'yolo_iou': [],
        'class_accuracy': defaultdict(list),
        'detection_count': defaultdict(int),
        'all_predictions': [],
        'all_ground_truths': []
    }
    
    return segmentation_metrics, detection_metrics

def evaluate_model(model, test_loader, criterion, device, class_names):
    """模型评估的主循环"""
    segmentation_metrics, detection_metrics = initialize_metrics()
    
    # 用于可视化的样本收集
    all_images = []
    all_true_masks = []
    all_pred_masks = []
    all_pred_boxes = []
    all_true_boxes = []
    
    print("开始评估...")
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(test_loader, desc="评估")):
            images = images.to(device, dtype=torch.float32)
            labels_segment = labels[0].to(device, dtype=torch.long)
            labels_yolo = labels[1].to(device, dtype=torch.float)
            
            # 模型推理
            pred_segment, pred_yolo = model(images)
            
            # 计算分割指标
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
                
                # 处理类别准确率
                if isinstance(class_acc, float):
                    detection_metrics['class_accuracy']['overall'].append(class_acc)
                elif isinstance(class_acc, dict):
                    for cls_name, acc_value in class_acc.items():
                        detection_metrics['class_accuracy'][cls_name].append(acc_value)
                        cls_idx = int(cls_name.split('_')[1]) if '_' in cls_name else 0
                        if acc_value > 0.5:
                            detection_metrics['detection_count'][cls_idx] += 1

                # 收集用于计算mAP的预测和真实框
                for b in range(pred_yolo.size(0)):
                    preds, gts = calculate_precision_recall(
                        pred_yolo[b].cpu().numpy(), 
                        labels_yolo[b].cpu().numpy(), 
                        iou_threshold=0.5, 
                        num_classes=5
                    )
                    detection_metrics['all_predictions'].extend(preds)
                    detection_metrics['all_ground_truths'].extend(gts)
            except Exception as e:
                print(f"计算目标检测指标出错: {str(e)}")
                print(f"错误类型: {type(e).__name__}")
                import traceback
                traceback.print_exc()
            
            # 保存样本用于可视化(最多保存10个batch)
            if idx < 10:
                for b in range(min(images.size(0), 2)):  # 每个batch最多保存2张图
                    img = images[b].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                    true_mask = labels_segment[b].cpu().numpy()
                    pred_mask = pred_labels[b].cpu().numpy()
                    
                    all_images.append(img)
                    all_true_masks.append(true_mask)
                    all_pred_masks.append(pred_mask)
                    all_true_boxes.append(labels_yolo[b].cpu().numpy())
                    all_pred_boxes.append(pred_yolo[b].cpu().numpy())
    
    return (
        segmentation_metrics, detection_metrics, 
        all_images, all_true_masks, all_pred_masks, 
        all_true_boxes, all_pred_boxes
    )

def calculate_final_metrics(segmentation_metrics, detection_metrics, class_names):
    """计算最终指标，包括平均值和mAP"""
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

    # 计算mAP (COCO格式评估指标)
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ap_dict, precision_dict, recall_dict = compute_ap_from_matches(
        detection_metrics['all_predictions'],
        detection_metrics['all_ground_truths'],
        iou_thresholds=iou_thresholds,
        num_classes=5
    )

    # 计算每个IoU阈值下的mAP
    map_50 = np.mean([ap_dict[0.5][cls] for cls in range(5) if cls in ap_dict[0.5]]) if 0.5 in ap_dict and ap_dict[0.5] else 0
    map_75 = np.mean([ap_dict[0.75][cls] for cls in range(5) if cls in ap_dict[0.75]]) if 0.75 in ap_dict and ap_dict[0.75] else 0
    
    # 计算mAP@[0.5:0.95] (COCO标准)
    maps = []
    for thresh in iou_thresholds:
        if thresh in ap_dict:
            cls_aps = [ap_dict[thresh][cls] for cls in range(5) if cls in ap_dict[thresh]]
            if cls_aps:
                maps.append(np.mean(cls_aps))
    map_all = np.mean(maps) if maps else 0
    
    results = {
        'avg_loss': avg_loss,
        'avg_miou': avg_miou,
        'avg_yolo_iou': avg_yolo_iou,
        'class_iou_avg': class_iou_avg,
        'class_det_acc': class_det_acc,
        'ap_dict': ap_dict,
        'map_50': map_50,
        'map_75': map_75,
        'map_all': map_all
    }
    
    return results

def print_results(results, class_names, detection_metrics):
    """打印评估结果"""
    print("\n=== 分割评估结果 ===")
    print(f"平均损失: {results['avg_loss']:.4f}")
    print(f"平均mIoU: {results['avg_miou']:.4f}")
    print("各类别IoU:")
    for cls in range(6):
        class_name = class_names[cls]
        iou_val = results['class_iou_avg'][cls] if cls in results['class_iou_avg'] else float('nan')
        print(f"  - {class_name}: {iou_val:.4f}")

    print("\n=== 目标检测评估结果 ===")
    print(f"平均YOLO IoU: {results['avg_yolo_iou']:.4f}")
    print(f"mAP@.5: {results['map_50']:.4f}")
    print(f"mAP@.75: {results['map_75']:.4f}")
    print(f"mAP@[.5:.95]: {results['map_all']:.4f}")
    print("各类别检测准确率:")
    for cls in range(1, 6):  # 跳过背景类
        class_name = class_names[cls]
        acc_val = results['class_det_acc'][cls] if cls in results['class_det_acc'] else float('nan')
        det_count = detection_metrics['detection_count'][cls]
        ap_50 = results['ap_dict'][0.5][cls-1] if 0.5 in results['ap_dict'] and cls-1 in results['ap_dict'][0.5] else float('nan')
        print(f"  - {class_name}: 准确率={acc_val:.4f}, AP@.5={ap_50:.4f} (检测到{det_count}个)")

def save_results_to_csv(results, class_names):
    """保存结果到CSV文件"""
    # 保存详细评估结果
    results_df = pd.DataFrame({
        'Class': [class_names[i] for i in range(6)],
        'Segmentation_IoU': [results['class_iou_avg'].get(i, float('nan')) for i in range(6)],
        'Detection_Accuracy': [results['class_det_acc'].get(i, float('nan')) for i in range(1, 6)] + [float('nan')]
    })
    results_df.to_csv('results/evaluation/detailed_results.csv', index=False)
    
    # 保存目标检测AP结果
    detection_df = pd.DataFrame({
        'Class': [class_names[i] for i in range(1, 6)],
        'Segmentation_IoU': [results['class_iou_avg'].get(i, float('nan')) for i in range(1, 6)],
        'Detection_Accuracy': [results['class_det_acc'].get(i, float('nan')) for i in range(1, 6)],
        'AP@.5': [results['ap_dict'][0.5].get(i-1, float('nan')) for i in range(1, 6)] if 0.5 in results['ap_dict'] else [float('nan')]*5,
        'AP@.75': [results['ap_dict'][0.75].get(i-1, float('nan')) for i in range(1, 6)] if 0.75 in results['ap_dict'] else [float('nan')]*5
    })
    detection_df.to_csv('results/evaluation/detection_results.csv', index=False)

from visualize import visualize_results
def main():
    # 创建保存结果的目录
    os.makedirs('results/evaluation', exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 类别映射
    class_names = {
        0: "Background",
        1: "Car",
        2: "Pedestrian",
        3: "Bicyclist",
        4: "MotorcycleScooter",
        5: "Truck_Bus"
    }

    # 加载模型
    print("正在加载模型...")
    model_path = 'results/deeplabmodelfullfinal.pth'
    model = torch.load(model_path, weights_only=False)
    model.eval()
    model.to(device)

    # 加载测试数据
    print("正在加载测试数据...")
    batch_size = 4
    test_dataset = Comp0249Dataset('data/CamVid', "val", scale=1, transform=None, target_transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 加载损失函数
    criterion = nn.CrossEntropyLoss()

    # 评估模型
    evaluation_results = evaluate_model(model, test_loader, criterion, device, class_names)
    segmentation_metrics, detection_metrics = evaluation_results[:2]
    all_images, all_true_masks, all_pred_masks, all_true_boxes, all_pred_boxes = evaluation_results[2:]

    # 计算最终指标
    results = calculate_final_metrics(segmentation_metrics, detection_metrics, class_names)
    
    # 打印结果
    print_results(results, class_names, detection_metrics)
    
    # 保存结果到CSV
    save_results_to_csv(results, class_names)
    
    # 可视化结果
    visualize_results(results, segmentation_metrics, class_names, 
                      all_images, all_true_masks, all_pred_masks, 
                      all_true_boxes, all_pred_boxes)
    
    print(f"\nmAP结果总结：\nmAP@.5={results['map_50']:.4f}, "
          f"mAP@.75={results['map_75']:.4f}, mAP@[.5:.95]={results['map_all']:.4f}")
    print("\n评估完成！结果已保存到results/evaluation/目录")

if __name__ == "__main__":
    main()
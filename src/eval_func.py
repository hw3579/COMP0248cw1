# filepath: c:\Users\A\Desktop\hw3579\project_jiaqiyao\src\eval_func.py
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import draw_the_box
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import tempfile
import os


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

def calculate_final_metrics(segmentation_metrics, detection_metrics, class_names, per_image_predictions, per_image_ground_truths):
    """计算最终指标，仅包含基本指标；不进行 mAP 评估"""
    # 1. 计算基本评估指标
    avg_loss = np.mean(segmentation_metrics['total_loss'])
    avg_miou = np.mean(segmentation_metrics['mean_iou'])
    avg_yolo_iou = np.mean(detection_metrics['yolo_iou'])

    # 2. 计算每个类别的平均 IoU 和检测准确率
    class_iou_avg = {cls: np.mean(vals) if vals else float('nan') 
                     for cls, vals in segmentation_metrics['class_iou'].items()}
    class_det_acc = {cls: np.mean(vals) if vals else float('nan') 
                     for cls, vals in detection_metrics['class_accuracy'].items()}

    # 删除了后续有关 mAP 和 AP 字典的代码，不再进行目标检测 mAP 评估

    return {
        'avg_loss': avg_loss,
        'avg_miou': avg_miou,
        'avg_yolo_iou': avg_yolo_iou,
        'class_iou_avg': class_iou_avg,
        'class_det_acc': class_det_acc
    }


def print_results(results, class_names, detection_metrics):
    """打印评估结果"""
    print("\n=== 分割评估结果 ===")
    print(f"平均损失: {results['avg_loss']:.4f}")
    print(f"平均mIoU: {results['avg_miou']:.4f}")
    print("各类别IoU:")
    for cls in range(6):
        class_name = class_names[cls]
        iou_val = results['class_iou_avg'].get(cls, float('nan'))
        print(f"  - {class_name}: {iou_val:.4f}")

    print("\n=== 目标检测评估结果 ===")
    print(f"平均YOLO IoU: {results['avg_yolo_iou']:.4f}")
    print("各类别检测准确率:")
    for cls in range(1, 6):  # 跳过背景类
        class_name = class_names[cls]
        acc_val = results['class_det_acc'].get(cls, float('nan'))
        det_count = detection_metrics['detection_count'][cls]
        print(f"  - {class_name}: 准确率={acc_val:.4f} (检测到{det_count}个)")

def save_results_to_csv(results, class_names):
    """保存结果到CSV文件"""
    results_df = pd.DataFrame({
        'Class': [class_names[i] for i in range(6)],
        'Segmentation_IoU': [results['class_iou_avg'].get(i, float('nan')) for i in range(6)],
        'Detection_Accuracy': [results['class_det_acc'].get(i, float('nan')) for i in range(1, 6)] + [float('nan')]
    })
    results_df.to_csv('results/evaluation/detailed_results.csv', index=False)

    detection_df = pd.DataFrame({
        'Class': [class_names[i] for i in range(1, 6)],
        'Segmentation_IoU': [results['class_iou_avg'].get(i, float('nan')) for i in range(1, 6)],
        'Detection_Accuracy': [results['class_det_acc'].get(i, float('nan')) for i in range(1, 6)]
    })
    detection_df.to_csv('results/evaluation/detection_results.csv', index=False)


def cxcywh_to_xy(boxes):
    """Convert center x, center y, width, height to top-left and right-bottom."""
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return np.stack([x1, y1, x2, y2], axis=-1)



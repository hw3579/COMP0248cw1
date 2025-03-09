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
    """Initialize evaluation metrics dictionary"""
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
    """Calculate final metrics, including only basic metrics; no mAP evaluation"""
    # 1. Calculate basic evaluation metrics
    avg_loss = np.mean(segmentation_metrics['total_loss'])
    avg_miou = np.mean(segmentation_metrics['mean_iou'])
    avg_yolo_iou = np.mean(detection_metrics['yolo_iou'])

    # 2. Calculate average IoU for each class and detection accuracy
    class_iou_avg = {cls: np.mean(vals) if vals else float('nan') 
                     for cls, vals in segmentation_metrics['class_iou'].items()}
    class_det_acc = {cls: np.mean(vals) if vals else float('nan') 
                     for cls, vals in detection_metrics['class_accuracy'].items()}

    # Removed subsequent code related to mAP and AP dictionaries, no longer performing object detection mAP evaluation

    return {
        'avg_loss': avg_loss,
        'avg_miou': avg_miou,
        'avg_yolo_iou': avg_yolo_iou,
        'class_iou_avg': class_iou_avg,
        'class_det_acc': class_det_acc
    }

def save_results_to_csv(results, class_names):
    """Save results to CSV"""
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
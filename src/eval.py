# filepath: c:\Users\A\Desktop\hw3579\project_jiaqiyao\src\eval.py
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


from train import compute_iou
from model import TotalDeepLabV3Plus
from dataloader import Comp0249Dataset
from utils import compute_iou_yolo, draw_the_box
from eval_func import initialize_metrics, calculate_final_metrics, save_results_to_csv
from visualize import visualize_results

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import tempfile

def evaluate_model(model, test_loader, criterion, device, class_names):
    """Main evaluation loop for the model"""
    segmentation_metrics, detection_metrics = initialize_metrics()
    
    # Collect samples for visualization
    all_images = []
    all_true_masks = []
    all_pred_masks = []
    all_pred_boxes = []
    all_true_boxes = []
    per_image_predictions = []
    per_image_ground_truths = []
    total_image_predictions = []
    total_image_ground_truths = []
    
    
    print("Starting evaluation...")
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device, dtype=torch.float32)
            labels_segment = labels[0].to(device, dtype=torch.long)
            labels_yolo = labels[1].to(device, dtype=torch.float)
            
            # Model inference
            pred_segment, pred_yolo = model(images)
            # pred_yolo = labels_yolo.clone()
            
            # Calculate segmentation metrics
            seg_loss = criterion(pred_segment, labels_segment)
            iou_dict, batch_mean_iou = compute_iou(pred_segment, labels_segment, num_classes=6)
            
            # Save segmentation metrics
            segmentation_metrics['total_loss'].append(seg_loss.item())
            segmentation_metrics['mean_iou'].append(batch_mean_iou)
            
            # Save IoU for each class
            for cls in range(6):
                class_key = f'class_{cls}'
                if class_key in iou_dict:
                    iou_value = iou_dict[class_key].item() if isinstance(iou_dict[class_key], torch.Tensor) else iou_dict[class_key]
                    if not np.isnan(iou_value):
                        segmentation_metrics['class_iou'][cls].append(iou_value)
            
            # Calculate predicted labels for confusion matrix
            pred_labels = torch.argmax(pred_segment, dim=1)
            
            # Update confusion matrix
            for b in range(labels_segment.size(0)):
                true_flat = labels_segment[b].cpu().numpy().flatten()
                pred_flat = pred_labels[b].cpu().numpy().flatten()
                cm = confusion_matrix(true_flat, pred_flat, labels=range(6))
                segmentation_metrics['confusion_matrix'] += cm
            
            # Calculate object detection metrics
            # try:
            if True:
                yolo_iou, class_acc = compute_iou_yolo(pred_yolo, labels_yolo)
                detection_metrics['yolo_iou'].append(yolo_iou)
                
                # Process class accuracy
                if isinstance(class_acc, float):
                    detection_metrics['class_accuracy']['overall'].append(class_acc)
                elif isinstance(class_acc, dict):
                    for cls_name, acc_value in class_acc.items():
                        detection_metrics['class_accuracy'][cls_name].append(acc_value)
                        cls_idx = int(cls_name.split('_')[1]) if '_' in cls_name else 0
                        if acc_value > 0.5:
                            detection_metrics['detection_count'][cls_idx] += 1

                # Collect predictions and ground truth boxes for mAP calculation
                for b in range(pred_yolo.size(0)):
                    img = images[b].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                    true_mask = labels_segment[b].cpu().numpy()
                    pred_mask = pred_labels[b].cpu().numpy()
                    
                    per_image_ground_truths.append(labels_yolo[b].cpu().numpy())
                    per_image_predictions.append(pred_yolo[b].cpu().numpy())
                  

            # Save samples for visualization (at most 10 batches)
            if idx < 10:
                for b in range(min(images.size(0), 2)):  # At most 2 images per batch
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
        all_true_boxes, all_pred_boxes,
        per_image_predictions, per_image_ground_truths  # New return values
    )


def main():
    # Create directory for saving results
    os.makedirs('results/evaluation', exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Class mapping
    class_names = {
        0: "Background",
        1: "Car",
        2: "Pedestrian",
        3: "Bicyclist",
        4: "MotorcycleScooter",
        5: "Truck_Bus"
    }

    # Load model
    print("Loading model...")
    model_path = 'results/deeplabmodelfullfinal.pth'
    model = torch.load(model_path, weights_only=False)
    model.eval()
    model.to(device)

    # Load test data
    print("Loading test data...")
    batch_size = 4
    test_dataset = Comp0249Dataset('data/CamVid', "val", scale=1, transform=None, target_transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Load loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate model
    evaluation_results = evaluate_model(model, test_loader, criterion, device, class_names)
    segmentation_metrics, detection_metrics = evaluation_results[:2]
    all_images, all_true_masks, all_pred_masks, all_true_boxes, all_pred_boxes = evaluation_results[2:7]
    per_image_predictions, per_image_ground_truths = evaluation_results[7:9]

    # Calculate final metrics
    results = calculate_final_metrics(segmentation_metrics, detection_metrics, class_names, per_image_predictions, per_image_ground_truths)  
    
    # Save results to CSV
    save_results_to_csv(results, class_names)
    
    # Visualize results
    visualize_results(results, segmentation_metrics, class_names, 
                      all_images, all_true_masks, all_pred_masks, 
                      all_true_boxes, all_pred_boxes)
    

if __name__ == "__main__":
    main()
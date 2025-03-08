from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from utils import draw_the_box

def visualize_results(results, segmentation_metrics, class_names, 
                      all_images, all_true_masks, all_pred_masks, 
                      all_true_boxes, all_pred_boxes):
    """Visualize evaluation results"""
    # 1. Confusion matrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(segmentation_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=[class_names[i] for i in range(6)],
                yticklabels=[class_names[i] for i in range(6)])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Segmentation Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/evaluation/confusion_matrix.png')
    plt.close()

    # 2. Segmentation results visualization
    num_samples = min(8, len(all_images))
    plt.figure(figsize=(15, 5*num_samples))

    for i in range(num_samples):
        # Original image
        plt.subplot(num_samples, 3, i*3+1)
        plt.imshow(all_images[i])
        plt.title(f"Sample {i+1} - Input Image")
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(num_samples, 3, i*3+2)
        plt.imshow(all_true_masks[i], cmap='tab20')
        plt.title(f"Sample {i+1} - Ground Truth Mask")
        plt.axis('off')
        
        # Predicted mask
        plt.subplot(num_samples, 3, i*3+3)
        plt.imshow(all_pred_masks[i], cmap='tab20')
        plt.title(f"Sample {i+1} - Predicted Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/evaluation/segmentation_results.png')
    plt.close()

    # 3. Object detection results visualization
    plt.figure(figsize=(15, 5*num_samples))

    for i in range(num_samples):
        # True bounding boxes
        plt.subplot(num_samples, 2, i*2+1)
        img_with_true_boxes = draw_the_box(all_images[i].copy(), all_true_boxes[i], nms_threshold=0.8)
        plt.imshow(img_with_true_boxes)
        plt.title(f"Sample {i+1} - Ground Truth Boxes")
        plt.axis('off')
        
        # Predicted bounding boxes
        plt.subplot(num_samples, 2, i*2+2)
        img_with_pred_boxes = draw_the_box(all_images[i].copy(), all_pred_boxes[i], nms_threshold=0.05)
        plt.imshow(img_with_pred_boxes)
        plt.title(f"Sample {i+1} - Predicted Boxes (NMS)")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/evaluation/detection_results.png')
    plt.close()

    # 4. Class performance bar chart
    plt.figure(figsize=(12, 6))
    classes = list(class_names.values())
    ious = [results['class_iou_avg'].get(i, 0) for i in range(6)]
    colors = ['#3498db' if i not in [4, 5] else '#e74c3c' for i in range(6)]  # Minority classes in red

    bars = plt.bar(classes, ious, color=colors)
    plt.axhline(y=results['avg_miou'], color='r', linestyle='--', 
                label=f'Average mIoU: {results["avg_miou"]:.4f}')
    plt.xlabel('Class')
    plt.ylabel('IoU')
    plt.title('IoU by Class')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)

    # Label values
    for bar, iou in zip(bars, ious):
        if not np.isnan(iou):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{iou:.3f}', ha='center', va='bottom')

    plt.legend()
    plt.tight_layout()
    plt.savefig('results/evaluation/class_performance.png')
    plt.close()

    # 5. Object detection AP visualization
    classes_det = list(class_names.values())[1:]  # Skip background class
    ap_values_50 = [results['ap_dict'][0.5].get(i-1, 0) for i in range(1, 6)] if 0.5 in results['ap_dict'] else [0]*5
    ap_values_75 = [results['ap_dict'][0.75].get(i-1, 0) for i in range(1, 6)] if 0.75 in results['ap_dict'] else [0]*5
    
    x = np.arange(len(classes_det))
    width = 0.35
    
    fig_ap, ax_ap = plt.subplots(figsize=(12, 6))
    rects1 = ax_ap.bar(x - width/2, ap_values_50, width, label='AP@.5')
    rects2 = ax_ap.bar(x + width/2, ap_values_75, width, label='AP@.75')
    
    ax_ap.set_ylabel('AP')
    ax_ap.set_title('AP by Class')
    ax_ap.set_xticks(x)
    ax_ap.set_xticklabels(classes_det)
    ax_ap.legend()
    
    ax_ap.bar_label(rects1, padding=3, fmt='%.3f')
    ax_ap.bar_label(rects2, padding=3, fmt='%.3f')
    
    fig_ap.tight_layout()
    plt.savefig('results/evaluation/detection_ap.png')
    plt.close()
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
    num_samples = min(2, len(all_images))
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
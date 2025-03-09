import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
from torchvision.io import read_image
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from dataloader import Comp0249Dataset
from tqdm import tqdm
import math

import datetime
from model import ResNetHead, ASPP, DeepLabV3PlusDecoder, ConvBlock, StageBlock1, StageBlock2, StageBlock3, StageBlock4, YOLOHead, TotalDeepLabV3Plus, StageBlockmid, StageBlock4_2
import platform
from torch.amp import autocast, GradScaler
from utils import segmentation_to_yolov3_1, yolo_loss, compute_iou_yolo
import os


import json 

def save_checkpoint(model, total_loss, total_acc, total_yolo_acc, epoch, best_loss, best_epoch, optimizer, reason="best"):
    """Save checkpoint and training data
    
    Args:
        model: The model to save
        total_loss, total_acc, total_yolo_acc: Training history records
        epoch: Current epoch
        best_loss, best_epoch: Best performance records
        optimizer: The optimizer
        reason: Reason for saving, 'best'/'early_stop'/'interrupted'
    
    Returns:
        tuple: (model_filename, data_filename)
    """
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_filename = f'results/deeplabmodelfullfinal_{reason}.pth'
    torch.save(model, model_filename)
    
    # Save training data
    data = {
        'loss': total_loss,
        'accuracy': total_acc,
        'yolo_accuracy': total_yolo_acc,  
        'last_epoch': epoch,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'save_time': timestamp,
        'reason': reason
    }
    
    # Save data to json file
    data_filename = f'results/deeplabmodeldatafinal_{reason}_{timestamp}.json'
    with open(data_filename, 'w') as f:
        json.dump(data, f)
    
    return model_filename, data_filename

def compute_iou(pred, labels, num_classes=6):
    """
    Calculate IoU between two label images (compute IoU for each class and average).
    
    Parameters:
        pred (Tensor): Model predictions (N, C, H, W)
        labels (Tensor): Ground truth labels (N, H, W), values 0 to num_classes-1
        num_classes (int): Number of semantic segmentation classes (including background)
    
    Returns:
        iou_dict (dict): IoU value for each class
        mean_iou (float): Average IoU
    """
    # Convert predictions to class labels
    pred = torch.argmax(pred, dim=1)  # (N, H, W)

    iou_dict = {}
    iou_list = []

    for cls in range(num_classes):
        intersection = ((pred == cls) & (labels == cls)).float().sum()
        union = ((pred == cls) | (labels == cls)).float().sum()

        if union == 0:
            iou = torch.tensor(float('nan'))  # Class does not exist in either image
        else:
            iou = intersection / union
        
        iou_dict[f'class_{cls}'] = iou
        if not torch.isnan(iou):
            iou_list.append(iou)

    # Calculate mean IoU (ignoring NaN classes)
    mean_iou = torch.tensor(iou_list).mean().item()
    
    return iou_dict, mean_iou

def test_compute_iou():
    # Generate two random label images
    torch.manual_seed(0)
    label1 = torch.randint(0, 6, (1, 480, 480))
    label2 = torch.randint(0, 6, (1, 480, 480))
    
    iou_dict, mean_iou = compute_iou(label1, label2, num_classes=6)
    print(iou_dict)
    print(mean_iou)


def calculate_class_weights(dataset, use_cache=True):
    """Calculate class weights for segmentation tasks, with caching support"""
    # Create cache directory
    cache_dir = 'data/CamVid/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'class_weights.pth')
    
    # Check if cache exists
    if use_cache and os.path.exists(cache_file):
        print(f"Loading class weights from cache: {cache_file}")
        class_weights = torch.load(cache_file)
        print(f"Loaded class weights: {class_weights}")
        return class_weights
    
    # No cache, recalculate
    print("Calculating class weights...")
    class_counts = {i: 0 for i in range(6)}  # Assuming 6 classes (0-5)
    
    for idx in tqdm(range(len(dataset)), desc="Analyzing class distribution"):
        _, [label, _] = dataset[idx]
        unique_classes, counts = torch.unique(label, return_counts=True)
        
        for cls, count in zip(unique_classes.tolist(), counts.tolist()):
            if cls in class_counts:
                class_counts[cls] += count
    
    print(f"Class distribution: {class_counts}")
    
    # Calculate weights (minority classes get higher weights)
    total_pixels = sum(class_counts.values())
    class_weights = torch.ones(6)  # Assuming 6 classes
    for cls, count in class_counts.items():
        if count > 0:
            # Use inverse proportion weighting and normalize
            class_weights[cls] = total_pixels / (len(class_counts) * count)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    print(f"Class weights: {class_weights}")
    
    # Save cache
    if use_cache:
        torch.save(class_weights, cache_file)
        print(f"Saved class weights cache: {cache_file}")
    
    return class_weights

if __name__ == "__main__":
    # Check for and remove existing exit flag file
    exit_file = "exit_training.txt"
    if os.path.exists(exit_file):
        os.remove(exit_file)
    print(f"During training, creating file '{exit_file}' will save the current model and exit training")

    # Define image augmentation strategy
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
    ])

    # Start training
    is_use_autoscale = True

    train_dataset = Comp0249Dataset('data/CamVid', "train", scale=1, transform=None, target_transform=None)

    # Configure dataloader according to platform and settings
    if is_use_autoscale:
        if platform.system() == 'Windows':
            train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0, pin_memory=True)
        if platform.system() == 'Linux':
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    else:
        if platform.system() == 'Windows':
            train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
        if platform.system() == 'Linux':
            train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=10)




    # model = TotalDeepLabV3Plus(num_classes=6, w=960, h=720)



    model = torch.load('results/deeplabmodelfullfinal_interrupted.pth', weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Slightly increased initial learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    total_loss = []
    total_acc = []
    num_epochs = 500

    criterion = nn.CrossEntropyLoss()  # Default will ignore invalid classes
    lambda_yolo = 0

    if is_use_autoscale:
        scaler = GradScaler()

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        loss_per_epoch = 0.0
        seg_loss_per_epoch = 0.0
        yolo_loss_per_epoch = 0.0
        acc_per_epoch = 0.0
        yolo_acc_per_epoch = 0.0  # YOLO accuracy tracking variable

        for images, labels in tqdm(train_loader, desc="Batches"):
            images = images.to(device, dtype=torch.float32)
            labels_segment = labels[0].to(device, dtype=torch.long)
            labels_yolo = labels[1].to(device, dtype=torch.float32)

            optimizer.zero_grad()
            if is_use_autoscale:
                with autocast(device_type=str(device)):
                    # Forward pass with modified loss function
                    pred, pred_yolo = model(images)

                    # pred, pred_yolo = images, labels_yolo

                    # b, h, w = labels_segment.size()
                    # num_classes = 6
                    # pred_segment = torch.zeros(b, num_classes, h, w, device=device)
                    
                    # for i in range(num_classes):
                    #     pred_segment[:, i, :, :] = (labels_segment == i).float() * 10.0

                    # pred = pred_segment

                    seg_loss = criterion(pred, labels_segment)
                    yolo_loss_val = yolo_loss(pred_yolo, labels_yolo, 8, 6, 1, 5, gamma=2.0, alpha=0.25)

                    lambda_yolo = seg_loss_per_epoch / (yolo_loss_per_epoch + 1e-8)
                    batch_total_loss = seg_loss + yolo_loss_val * lambda_yolo
                
                # Use correct loss for scaling and backpropagation
                scaler.scale(batch_total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred, pred_yolo = model(images)
                seg_loss = criterion(pred, labels_segment)
                yolo_loss_val = yolo_loss(pred_yolo, labels_yolo, 8, 6, 1, 5)

                lambda_yolo = seg_loss_per_epoch / (yolo_loss_per_epoch + 1e-8)
                batch_total_loss = seg_loss + yolo_loss_val * lambda_yolo
                batch_total_loss.backward()
                optimizer.step()

            # Accumulate losses
            loss_per_epoch += batch_total_loss.item()
            seg_loss_per_epoch += seg_loss.item()
            yolo_loss_per_epoch += yolo_loss_val.item()

            # Calculate accuracy
            _, batch_acc = compute_iou(pred, labels_segment)
            acc_per_epoch += batch_acc

            # Calculate YOLO accuracy
            with torch.no_grad():
                batch_yolo_iou_, batch_yolo_iou = compute_iou_yolo(pred_yolo, labels_yolo)
                yolo_acc_per_epoch += batch_yolo_iou

        # Calculate average losses and accuracies per epoch
        seg_loss_per_epoch /= len(train_loader)
        yolo_loss_per_epoch /= len(train_loader)
        loss_per_epoch /= len(train_loader)
        acc_per_epoch /= len(train_loader)
        yolo_acc_per_epoch /= len(train_loader)
        
        print(f"segloss:{seg_loss_per_epoch:.4f}, yololoss:{yolo_loss_per_epoch:.4f}")
        print(f"Epoch: {epoch}, Loss: {loss_per_epoch:.4f}, Acc: {acc_per_epoch:.4f}, YOLO Class Acc: {yolo_acc_per_epoch:.4f}")

        total_loss.append(loss_per_epoch)
        total_acc.append(acc_per_epoch)
        
        # Record YOLO accuracy
        if 'total_yolo_acc' not in locals():
            total_yolo_acc = []
        total_yolo_acc.append(yolo_acc_per_epoch)

        # Learning rate adjustment
        scheduler.step(loss_per_epoch)

        # Early stopping configuration
        patience = 10  # Number of epochs with no improvement before stopping
        min_delta = 1e-5  # Minimum threshold for improvement
        min_epochs_before_earlystop = 125  # Minimum epochs before checking early stopping

        # Initialize early stopping variables
        # Initialize early stopping variables
        if epoch == 0:
            best_loss = loss_per_epoch
            best_epoch = 0
            counter = 0
        # Check if loss has improved
        elif loss_per_epoch < best_loss - min_delta:
            best_loss = loss_per_epoch
            best_epoch = epoch
            counter = 0
            
            model_filename, _ = save_checkpoint(model, total_loss, total_acc, total_yolo_acc, 
                                            epoch, best_loss, best_epoch, optimizer, reason="best")
            print(f"Saved best model: {model_filename}")
        else:
            # Only increase counter after minimum training epochs
            if epoch >= min_epochs_before_earlystop:
                counter += 1
                print(f"Early stopping counter: {counter}/{patience} (active after {min_epochs_before_earlystop} epochs)")
            else:
                print(f"Early stopping not active yet, will activate after {min_epochs_before_earlystop} epochs")
            
        # If no improvement for 'patience' epochs and after minimum training epochs, stop training
        if counter >= patience and epoch >= min_epochs_before_earlystop:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with loss {best_loss:.4f}")
            model_filename, data_filename = save_checkpoint(model, total_loss, total_acc, total_yolo_acc, 
                                                    epoch, best_loss, best_epoch, optimizer, reason="early_stop")
            print(f"Saved early stopping data:\n- {model_filename}\n- {data_filename}")
            break

        # Check for exit signal
        if os.path.exists(exit_file):
            print("\nExit signal detected, saving model and training data...")
            
            model_filename, data_filename = save_checkpoint(model, total_loss, total_acc, total_yolo_acc, 
                                                    epoch, best_loss, best_epoch, optimizer, reason="interrupted")
            
            os.remove(exit_file)
            print(f"Saved interrupted training data:\n- {model_filename}\n- {data_filename}")
            print("Training exited")
            break
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import TotalModel#, segmentation_loss, 
from train import yolo_loss_func, yolo_accuracy
from dataloader import Comp0249Dataset
from utils import cx_cy_to_corners
import numpy as np
from utils import draw_the_box
from train_deeplabv3 import compute_iou
from model import ConvBlock, ResNetHead, ASPP, DeepLabV3PlusDecoder, TotalDeepLabV3Plus, StageBlock1, StageBlock2, StageBlock3, StageBlock4, YOLOHead
import torch.nn as nn

from utils import compute_iou_yolo

model = torch.load('results/deeplabmodelfullfinal.pth', weights_only=False)


model.eval()  # 切换到评估模式
total_loss = []
total_acc = []
total_yolo_iou = []

# 在测试集上进行评估
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#load test data
test_dataset = Comp0249Dataset('data/CamVid', "val", scale=1, transform=None, target_transform=None)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

is_plot = True

import cv2
import numpy as np
import matplotlib.pyplot as plt


criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device, dtype=torch.float32)
        labels_seg_with_batch = labels[0].to(device, dtype=torch.long)
        labels_yolo_with_batch = labels[1].to(device, dtype=torch.float)
        # output = labels
        output, output_yolo = model(images)

        loss = criterion(output, labels_seg_with_batch)
        total_loss.append(loss.item())
        _, batch_acc = compute_iou(output, labels_seg_with_batch, 6)
        total_acc.append(batch_acc)
        print('loss', loss.item(), 'acc', batch_acc)

        yolo_iou, class_acc = compute_iou_yolo(output_yolo, labels_yolo_with_batch)
        print('yolo_iou', yolo_iou)
        total_yolo_iou.append(yolo_iou)


        if is_plot:
            for _batch in range(len(images)):

                show_image = images[_batch].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                labels_seg = labels_seg_with_batch[_batch].cpu().numpy()
                labels_yolo = labels_yolo_with_batch[_batch].cpu().numpy()

                pred_labels = torch.argmax(output[_batch], dim=0).cpu().numpy()

                plt.figure(figsize=(10, 10))
                plt.subplot(2, 2, 1)
                plt.imshow(show_image)
                plt.title('Input Image')
                plt.subplot(2, 2, 2)
                plt.imshow(labels_seg, cmap='gray')
                plt.title('Ground Truth')
                plt.subplot(2, 2, 3)
                plt.imshow(pred_labels, cmap='gray')
                plt.title('Prediction')
                plt.subplot(2, 2, 4)
                plt.imshow(draw_the_box(show_image, pred_labels))
                plt.title('Prediction with Box')
                plt.show()

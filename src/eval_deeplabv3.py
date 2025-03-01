import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import TotalModel#, segmentation_loss, 
from train import yolo_loss_func, yolo_accuracy
from dataloader import Comp0249Dataset, Comp0249DatasetYolo, cx_cy_to_corners
import numpy as np
from utils import draw_the_box
from train_deeplabv3 import compute_iou
from train_deeplabv3 import TotalDeepLabV3Plus, Backbone, ASPP, Bottleneck, DeepLabV3PlusDecoder
import torch.nn as nn

model = torch.load('results/deeplabmodelfull.pth', weights_only=False)

model.eval()  # 切换到评估模式
total_loss = []
total_acc = []

# 在测试集上进行评估
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#load test data
test_dataset = Comp0249Dataset('data/CamVid', "val", scale=1, transform=None, target_transform=None)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

is_plot = True

import cv2
import numpy as np


criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        # output = labels
        output = model(images)

        loss = criterion(output, labels)
        total_loss.append(loss.item())
        _, batch_acc = compute_iou(output, labels, 6)
        total_acc.append(batch_acc)
        print('loss', loss.item(), 'acc', batch_acc)

        if is_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 4, 1)
            show_image = images[0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            labels = labels.cpu().numpy()
            pred_labels = torch.argmax(output, dim=1).cpu().numpy()
            plt.imshow(draw_the_box(show_image, labels[0]))
            plt.title('Input Image')
            plt.subplot(1, 4, 2)
            plt.imshow(labels[0], cmap='gray')
            plt.title('Ground Truth')
            plt.subplot(1, 4, 3)
            plt.imshow(pred_labels[0], cmap='gray')
            plt.title('Prediction')
            plt.subplot(1, 4, 4)
            plt.imshow(draw_the_box(show_image, pred_labels[0]))
            plt.show()
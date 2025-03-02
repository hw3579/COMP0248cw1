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
import matplotlib.pyplot as plt
import os
import pandas as pd

from utils import segmentation_to_yolo
from utils import segmentation_to_yolov3
# print('Using PyTorch version:', torch.__version__)
# if torch.cuda.is_available():
#     print('Using GPU, device name:', torch.cuda.get_device_name(0))
#     device = torch.device('cuda')
# else:
#     print('No GPU found, using CPU instead.')
#     device = torch.device('cpu')

def filter_classes(classes : pd.DataFrame):
    selected_classes = {
        "Car": 1,
        "Pedestrian": 2,
        "Bicyclist": 3,
        "MotorcycleScooter": 4,
        "Truck_Bus": 5
    }
    classes['name'] = classes['name'].map(lambda x: selected_classes.get(x, 0))
    return classes

# class Comp0249Dataset(Dataset):
#     '''
#     Dataloader for the Comp0249 dataset
#     input:  dir - path to the dataset
#             classes - 'train' or 'val' or 'test'
#             scale - scale of the image
#             transform - image transformation
#             target_transform - label transformation
#             is_filter_classes - whether to filter the classes to only the ones we are interested in

#     '''
#     def __init__(self, dir: str, classes: str, scale=1, transform=None, target_transform=None, is_filter_classes=True):
        
#         self.dir = os.path.join(dir, classes)
#         self.dir_labels = os.path.join(dir, classes + '_labels')

#         self.class_dict = pd.read_csv(os.path.join(dir, 'class_dict.csv'))
#         self.class_dict['new_col'] = range(1, len(self.class_dict) + 1) #add index column

#         if is_filter_classes:
#             self.class_dict = filter_classes(self.class_dict)
#         self.class_dict = self.class_dict.values.tolist()

#         self.images = list(sorted(os.listdir(self.dir)))
#         self.images_labels = list(sorted(os.listdir(self.dir_labels)))

#         self.scale = scale
#         self.transform = transform
#         self.target_transform = target_transform



#     def __len__(self):
#         return len(self.images_labels)

#     def __getitem__(self, idx):
#         '''
#         input: idx - index of the image
#         output: image, label
#                 image - tensor of the image, dimensions (3, H, W)
#                 label - tensor of the label, dimensions (H, W)
        
        
#         '''

#         image = read_image(os.path.join(self.dir + "/" + self.images[idx]))
#         label = read_image(os.path.join(self.dir_labels + "/" + self.images_labels[idx]))

#         # image = Image.open(os.path.join(self.dir + "/" + self.images[idx])).convert("RGB")
#         # label = Image.open(os.path.join(self.dir_labels + "/" + self.images_labels[idx])).convert("L")

#         # smaller image for faster training (1/10) 720x960 -> 72x96
#         _, h, w = image.shape
#         resize = transforms.Resize((h // self.scale, w // self.scale))
#         image = resize(image)
#         label = resize(label)
#         _, h, w = image.shape
#         label_gray = torch.zeros(h, w, dtype=torch.uint8)

#         for item in self.class_dict:
#             # for hx in range(h):
#             #     for wx in range(w):
#             #         if (label[:, hx , wx] == torch.Tensor(item[1:4])).to(label.dtype).all():
#             #             label_gray[hx , wx] = item[0]

#             # item[0] 为类别编号，item[1:4] 为颜色，比如 [64,128,64]
#             color = torch.tensor(item[1:4], dtype=label.dtype).view(3, 1, 1)
#             # 比较整个 label 得到 (3, h, w) 的布尔张量，all(dim=0) 得到 (h, w) 掩码
#             mask = (label == color).all(dim=0)
#             # 对满足条件的像素赋予类别编号
#             label_gray[mask] = item[0]
       
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label_gray

#     def getitem(self, idx):
#         return self.__getitem__(idx)
    


class Comp0249Dataset(Dataset):
    '''
    Dataloader for the Comp0249 dataset
    input:  dir - path to the dataset
            classes - 'train' or 'val' or 'test'
            transform - image transformation
            scale - scale of the image
            target_transform - label transformation
            is_filter_classes - whether to filter the classes to only the ones we are interested in

    '''
    def __init__(self, dir: str, classes: str, scale = 1, transform=None, target_transform=None, is_filter_classes=True, version="binary"):

        self.dir = os.path.join(dir, classes)
        self.dir_labels = os.path.join(dir, classes + '_labels')

        self.class_dict = pd.read_csv(os.path.join(dir, 'class_dict.csv'))
        self.class_dict['new_col'] = range(1, len(self.class_dict) + 1) #add index column

        if is_filter_classes:
            self.class_dict = filter_classes(self.class_dict)
        self.class_dict = self.class_dict.values.tolist()

        self.images = list(sorted(os.listdir(self.dir)))
        self.images_labels = list(sorted(os.listdir(self.dir_labels)))
        

        self.scale = scale
        self.transform = transform
        self.target_transform = target_transform
        self.version = version



    def __len__(self):
        return len(self.images_labels)

    def __getitem__(self, idx):
        '''
        input: idx - index of the image
        output: image, label
                image - tensor of the image, dimensions (3, H, W)
                label - tensor of the label, dimensions (H, W)
        
        
        '''

        image = read_image(os.path.join(self.dir + "/" + self.images[idx]))
        label = read_image(os.path.join(self.dir_labels + "/" + self.images_labels[idx]))

        # smaller image for faster training (1/10) 720x960 -> 72x96
        _, h, w = image.shape
        resize = transforms.Resize((h // self.scale, w // self.scale))
        image = resize(image)
        label = resize(label)
        _, h, w = image.shape
        label_gray = torch.zeros(h, w, dtype=torch.uint8)

        for item in self.class_dict:
            # for hx in range(h):
            #     for wx in range(w):
            #         if (label[:, hx , wx] == torch.Tensor(item[1:4])).to(label.dtype).all():
            #             label_gray[hx , wx] = item[0]

            # item[0] 为类别编号，item[1:4] 为颜色，比如 [64,128,64]
            color = torch.tensor(item[1:4], dtype=label.dtype).view(3, 1, 1)
            # 比较整个 label 得到 (3, h, w) 的布尔张量，all(dim=0) 得到 (h, w) 掩码
            mask = (label == color).all(dim=0)
            # 对满足条件的像素赋予类别编号
            label_gray[mask] = item[0]    

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        if self.version == "binary":
            yolo_label = label_gray  
           
        elif self.version == "yolov1":
            yolo_label = segmentation_to_yolo(label_gray, S=7, num_classes=5, B=2, scale=self.scale)
        elif self.version == "yolov3":
            yolo_label = segmentation_to_yolov3(label_gray, w, h, num_classes=5, B=2, scale=self.scale)

        return image, yolo_label

    def getitem(self, idx):
        return self.__getitem__(idx)

import cv2

def cx_cy_to_corners(cx, cy, w, h):
    '''
    input: cx, cy, w, h - the center x, center y, width, and height of the box
    output: x1, y1, x2, y2 - the coordinates of the top left and bottom right corners of the box
    '''
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return x1, y1, x2, y2

def draw_the_yolo_label(image, yolo_label):
    '''
    input: image - the image to draw the box on
                    yolo_label: torch.Tensor, 尺寸 (S, S, num_classes + B*5)
                    格式为：前 num_classes 维：one-hot 目标类别；
                            后面 B*5 维，每个边框组成 [cx, cy, w, h, confidence]
    output: image - the image with the box drawn on it (H, W)
    '''
    H, W, _ = image.shape
    class_yolo = yolo_label[..., :5]
    bbox_yolo = yolo_label[..., 5:]
    for i in range(5):
        # 得到当前类别的掩码，类型为 bool
        mask = (class_yolo[..., i] > 0)
        # 如果该类别没有像素，则跳过
        if np.count_nonzero(mask) == 0:
            continue

        # 获得该类别像素的行和列索引
        rows, cols = np.where(mask)
        class_position = np.hstack((rows.reshape(-1, 1), cols.reshape(-1, 1)))

        for _ in range(len(class_position)):
            Sx, Sy = class_position[_]
            left = Sy * W // 7
            top = Sx * H // 7
            selected_yolo = bbox_yolo[Sx, Sy, :]

            position = selected_yolo[..., :4].numpy().flatten()

            position[0], position[1], position[2], position[3] = cx_cy_to_corners(*position)

            position = position * np.array([W, H, W, H])
            position = position.astype(np.int32)
            # 保证 image 内存连续
            image = np.ascontiguousarray(image)
            # 绘制矩形框
            image = cv2.rectangle(image, (position[0], position[1]), (position[2], position[3]), (255, 25*i, 255), 2)
        
    return image


from utils import draw_the_box
if __name__ == '__main__':
    # dataset = Comp0249Dataset('data/CamVid', "train", scale=1)
    # dataset.getitem(0)
    # # print(dataset[0])
    # ax, pl = plt.subplots(1, 2)
    # pl[0].imshow(dataset[0][0].permute(1, 2, 0))
    # pl[1].imshow(dataset[0][1], cmap='gray')

    # plt.show()

    # dataset = Comp0249DatasetYolo('data/CamVid', "train", scale=1)
    # dataset.getitem(0)
    # # print(dataset[0])
    # ax, pl = plt.subplots(1, 2)
    # image = dataset[0][0].permute(1, 2, 0)
    # label = dataset[0][1]
    # pl[0].imshow(image)
    # pl[1].imshow(draw_the_yolo_label(image, label))
    # plt.show()

    dataset = Comp0249Dataset('data/CamVid', "train", scale=1, version="yolov3")
    dataset.getitem(0)
    # print(dataset[0])
    ax, pl = plt.subplots(1, 2)
    image = dataset[0][0].permute(1, 2, 0)
    label = dataset[0][1][0]
    pl[0].imshow(image)
    pl[1].imshow(draw_the_yolo_label(image, label))
    plt.show()




    test_dataset = Comp0249Dataset('data/CamVid', "train")
    image, label = test_dataset.getitem(0)
    image = image.numpy().transpose(1, 2, 0).astype(np.uint8)
    label = label.numpy()
    image = draw_the_box(image, label)
    # automatically resize window
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('label', cv2.WINDOW_NORMAL)
    # large the image
    h, w, _ = image.shape
    cv2.resizeWindow('image', 960, 720)
    cv2.resizeWindow('label', 960, 720)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('image', image)
    cv2.imshow('label', label*20)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

class Comp0249Dataset(Dataset):
    '''
    Dataloader for the Comp0249 dataset
    input:  dir - path to the dataset
            classes - 'train' or 'val' or 'test'
            transform - image transformation
            target_transform - label transformation
            is_filter_classes - whether to filter the classes to only the ones we are interested in

    '''
    def __init__(self, dir: str, classes: str, transform=None, target_transform=None, is_filter_classes=True):

        self.dir = os.path.join(dir, classes)
        self.dir_labels = os.path.join(dir, classes + '_labels')

        self.class_dict = pd.read_csv(os.path.join(dir, 'class_dict.csv'))
        self.class_dict['new_col'] = range(1, len(self.class_dict) + 1) #add index column

        if is_filter_classes:
            self.class_dict = filter_classes(self.class_dict)
        self.class_dict = self.class_dict.values.tolist()

        self.images = list(sorted(os.listdir(self.dir)))
        self.images_labels = list(sorted(os.listdir(self.dir_labels)))

        self.transform = transform
        self.target_transform = target_transform



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

        # image = Image.open(os.path.join(self.dir + "/" + self.images[idx])).convert("RGB")
        # label = Image.open(os.path.join(self.dir_labels + "/" + self.images_labels[idx])).convert("L")

        # smaller image for faster training (1/10) 720x960 -> 72x96
        _, h, w = image.shape
        resize = transforms.Resize((h // 10, w // 10))
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
        return image, label_gray

    def testgetitem(self, idx):
        return self.__getitem__(idx)

if __name__ == '__main__':
    dataset = Comp0249Dataset('data/CamVid', "train")
    dataset.testgetitem(0)
    # print(dataset[0])
    ax, pl = plt.subplots(1, 2)
    pl[0].imshow(dataset[0][0].permute(1, 2, 0))
    pl[1].imshow(dataset[0][1], cmap='gray')

    plt.show()
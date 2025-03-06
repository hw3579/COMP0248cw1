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
from utils import segmentation_to_yolov3, segmentation_to_yolov3_1
from utils import draw_the_yolo_label

import random
import os.path as osp
import json
from tqdm import tqdm


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
            scale - scale of the image
            target_transform - label transformation
            is_filter_classes - whether to filter the classes to only the ones we are interested in

    '''
    def __init__(self, dir: str, classes: str, scale = 1, transform=None, target_transform=None, is_filter_classes=True, version="binary", use_cache=True):

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

        # 新增缓存相关参数
        self.use_cache = use_cache
        self.cache_dir = osp.join(dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.rare_class_cache_file = osp.join(self.cache_dir, f'{classes}_rare_classes_{scale}.json')
        self.class_dist_cache_file = osp.join(self.cache_dir, f'{classes}_class_dist_{scale}.json')
        
        # 加载或创建少数类缓存
        self.rare_class_images = self._load_or_create_rare_class_cache()
    
    def _load_or_create_rare_class_cache(self):
        """加载或创建少数类图像缓存"""
        if self.use_cache and osp.exists(self.rare_class_cache_file):
            print(f"正在加载少数类缓存: {self.rare_class_cache_file}")
            with open(self.rare_class_cache_file, 'r') as f:
                return json.load(f)
        else:
            print("正在分析数据集以识别少数类图像...")
            rare_class_images = {}
            for idx in tqdm(range(len(self.images)), desc="分析少数类"):
                image_name = self.images[idx]
                image_path = osp.join(self.dir, image_name)
                label_path = osp.join(self.dir_labels, self.images_labels[idx])
                
                # 读取并处理标签以确定类别
                label = read_image(label_path)
                _, h, w = label.shape
                resize = transforms.Resize((h // self.scale, w // self.scale))
                label = resize(label)
                label_gray = torch.zeros(h // self.scale, w // self.scale, dtype=torch.uint8)
                
                # 应用类别映射
                for item in self.class_dict:
                    color = torch.tensor(item[1:4], dtype=label.dtype).view(3, 1, 1)
                    mask = (label == color).all(dim=0)
                    label_gray[mask] = item[0]
                
                # 检查是否包含少数类
                unique_classes = torch.unique(label_gray).tolist()
                rare_classes = [cls for cls in unique_classes if cls in [4, 5]]
                
                if rare_classes:
                    rare_class_images[image_name] = {
                        "rare_classes": rare_classes,
                        "all_classes": unique_classes
                    }
            
            # 保存缓存
            if self.use_cache:
                with open(self.rare_class_cache_file, 'w') as f:
                    json.dump(rare_class_images, f)
                print(f"已保存少数类缓存: {self.rare_class_cache_file}")
            
            return rare_class_images

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

        # 检查是否包含少数类
        unique_classes = torch.unique(label_gray)
        contains_rare_class = any(cls in [4, 5] for cls in unique_classes)  
        
        # 使用缓存判断是否为少数类样本
        image_name = self.images[idx]
        contains_rare_class = image_name in self.rare_class_images
        
        # 对包含少数类的样本进行更强的数据增强
        if contains_rare_class:
            # print(f"应用少数类增强: {image_name}, 类别: {self.rare_class_images[image_name]['rare_classes']}")
            
            # 数据增强逻辑保持不变
            seed = torch.randint(0, 2**32, (1,)).item()
            
            torch.manual_seed(seed)
            random.seed(seed)
            image_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ])
            image = image_transform(image)
            
            torch.manual_seed(seed)
            random.seed(seed)
            label_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5)
            ])
            label_gray = label_transform(label_gray.unsqueeze(0)).squeeze(0)


        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        if self.version == "binary":
            binary_label = label_gray  
            yolo_label = segmentation_to_yolov3_1(label_gray, Sx = 8, Sy = 6 , num_classes=5, B=1, scale=1)

           
        # elif self.version == "yolov1":
        #     yolo_label = segmentation_to_yolo(label_gray, S=7, num_classes=5, B=2, scale=self.scale)
        # elif self.version == "yolov3":
        #     yolo_label = segmentation_to_yolov3(label_gray, w, h, num_classes=5, B=1, scale=self.scale)

        return image, [binary_label, yolo_label]

    def getitem(self, idx):
        return self.__getitem__(idx)
    



from utils import draw_the_box
if __name__ == '__main__':

    benchmark_dataloader = 0
    test_dataloader = 1




    if benchmark_dataloader:
        import time
        from torch.utils.data import DataLoader

        def benchmark_dataloader(num_workers):
            dataset = Comp0249Dataset('data/CamVid', "train", scale=1, version="yolov1")
            train_loader = DataLoader(dataset, batch_size=32, num_workers=num_workers)
            start = time.time()
            for _ in range(5):  # 运行5个 batch
                for batch in train_loader:
                    pass  # 仅测试数据加载速度
            end = time.time()
            return end - start

        for nw in range(3,6):
            time_taken = benchmark_dataloader(nw)
            print(f"num_workers={nw}, load time={time_taken:.4f}s")

        '''
        debug mode:
        num_workers=1, load time=423.4714s
        num_workers=2, load time=282.9612s
        num_workers=3, load time=198.7757s
        num_workers=4, load time=185.6829s
        num_workers=5, load time=202.7755s


        '''

    if test_dataloader:
        dataset = Comp0249Dataset('data/CamVid', "train", scale=1)
        dataset.getitem(0)
        # print(dataset[0])
        fig, pl = plt.subplots(2, 2)
        pl[0, 0].imshow(dataset[0][0].permute(1, 2, 0))
        pl[0, 1].imshow(dataset[0][1][0], cmap='gray')
        print(dataset[0][1][1].shape)

        # dataset_yolov1 = Comp0249Dataset('data/CamVid', "train", scale=1, version="yolov1")
        # dataset_yolov1.getitem(0)
        # image_yolov1 = dataset_yolov1[0][0].permute(1, 2, 0)
        # label_yolov1 = dataset_yolov1[0][1]
        # pl[1, 0].imshow(draw_the_yolo_label(image_yolov1, label_yolov1))

        # dataset_yolov3 = Comp0249Dataset('data/CamVid', "train", scale=1, version="yolov3")
        # dataset_yolov3.getitem(0)
        # image_yolov3 = dataset_yolov3[0][0].permute(1, 2, 0)
        # label_yolov3 = dataset_yolov3[0][1][2]
        # pl[1, 1].imshow(draw_the_yolo_label(image_yolov3, label_yolov3))

        plt.savefig('fig/data.png')
        plt.show()





        # test_dataset = Comp0249Dataset('data/CamVid', "train")
        # image, label = test_dataset.getitem(0)
        # image = image.numpy().transpose(1, 2, 0).astype(np.uint8)
        # label = label.numpy()
        # image = draw_the_box(image, label)
        # # automatically resize window
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('label', cv2.WINDOW_NORMAL)
        # # large the image
        # h, w, _ = image.shape
        # cv2.resizeWindow('image', 960, 720)
        # cv2.resizeWindow('label', 960, 720)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imshow('image', image)
        # cv2.imshow('label', label*20)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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
# 假设 test_loader 已经构建好

# 加载模型
# model = TotalModel(3, 6)
# model.load_state_dict(torch.load("results/model.pth")) # 加载模型参数

model = torch.load('results/full_model.pth', weights_only=False)

model.eval()  # 切换到评估模式
total_loss = []
total_acc = []

# 在测试集上进行评估
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#load test data
test_dataset = Comp0249Dataset('data/CamVid', "val", scale=1, transform=None, target_transform=None, version="yolov1")
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

is_plot = True

import cv2
import numpy as np


selected_classes = {
    "Car": 1,
    "Pedestrian": 2,
    "Bicyclist": 3,
    "MotorcycleScooter": 4,
    "Truck_Bus": 5
}
# 将 selected_classes 反转：数字 -> 类别名称
rev_selected_classes = {v: k for k, v in selected_classes.items()}

class_threshold = 0.5
confidence_threshold = 0.5

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float)
        # output = labels
        output = model(images)

        loss = yolo_loss_func(output, labels, model.S, model.B, model.C)
        total_loss.append(loss.item())
        batch_acc = yolo_accuracy(output, labels, C=model.C)
        total_acc.append(batch_acc.item())

        print('loss', loss.item(), 'acc', batch_acc.item())
        for _b in range(len(images)):
            position_class = output[_b,:,:,:5]
            position_xywh_bbox1 = output[_b,:,:,5:10]
            position_xywh_bbox2 = output[_b,:,:,10:15]  
            image = None         

            mask = position_class > class_threshold # [H,W,5]的布尔值
            indices = torch.nonzero(mask, as_tuple=False) # [N,3]的索引值，N为非零元素的个数

            for _i, _j, _c in indices:
                            _c = _c.item()
            # for _i in range(position_class.shape[0]):
            #     for _j in range(position_class.shape[1]):
            #         for _c in range(model.C):
            #             if position_class[_i][_j][_c] >0.5:
                            if position_xywh_bbox1[_i][_j][4] > position_xywh_bbox2[_i][_j][4]:
                                bbox_better = position_xywh_bbox1[_i][_j]
                            else:
                                bbox_better = position_xywh_bbox2[_i][_j]
                            
                            if bbox_better[4] > 0.5:
                                bbox_better = bbox_better[:4].cpu().numpy().flatten()

                                bbox_better[0] = (_j + bbox_better[0]) /model.S
                                bbox_better[1] = (_i + bbox_better[1]) /model.S


                                bbox_better[0], bbox_better[1], bbox_better[2], bbox_better[3] = cx_cy_to_corners(*bbox_better)
                                position = bbox_better * np.array([model.w, model.h, model.w, model.h])
                                position = position.astype(np.int32)
                                # 保证 image 内存连续
                                image = np.ascontiguousarray(images[_b].cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
                                # 绘制矩形框
                                image = cv2.rectangle(image, (position[0], position[1]), (position[2], position[3]), (255, 25*_c, 255), 2)
                                # 添加文字（例如：“目标”或者你需要的类别信息）
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                class_id = _c + 1
                                text = rev_selected_classes.get(class_id, f"id:{class_id}")  # 如果找不到，就显示 id

                                # 将文字放置在矩形框上方，如果超出边界，则放在矩形框内部
                                text_position = (position[0], position[1] - 10 if position[1] - 10 > 0 else position[1] + 20)
                                image = cv2.putText(image, text, text_position, font, 0.5, (255, 25*_b, 255), 2)

            if image is not None:
                plt.imshow(image)
                plt.show()
                        


import json
result = {
    "loss": total_loss,
    "accuracy": total_acc
}
with open('results/evaluation.json', 'w') as f:
    json.dump(result, f)

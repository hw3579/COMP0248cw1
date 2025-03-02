import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from yolov3 import CBL, ResUnit, ResUnitX, Yolov3
from yolov3 import yolo_loss_funcv3, yolo_accuracy_v3
from dataloader import Comp0249Dataset, cx_cy_to_corners
import numpy as np
from utils import draw_the_box

model = torch.load('results/full_model_yolov3.pth', weights_only=False)

model.eval()  # 切换到评估模式
total_loss = []
total_acc = []

# 在测试集上进行评估
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#load test data
test_dataset = Comp0249Dataset('data/CamVid', "val", scale=1, transform=None, target_transform=None, version="yolov3")
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=0)

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

w, h = 960, 720
model.w, model.h = w, h
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device, dtype=torch.float32)
        for i in range(len(labels)):
            labels[i] = labels[i].to(device, dtype=torch.float32)
        outputs = labels
        # outputs = model(images)

        loss = yolo_loss_funcv3(outputs, labels, w/32, h/32, C=5)
        total_loss.append(loss.item())
        batch_acc = yolo_accuracy_v3(outputs, labels, C=5)
        total_acc.append(batch_acc.item())

        print('loss', loss.item(), 'acc', batch_acc.item())
        for _cell in range(len(outputs)):
            output = outputs[_cell]
            for _b in range(len(images)):
                position_class = output[_b,:,:,:5]
                position_xywh_bbox1 = output[_b,:,:,5:10]
                position_xywh_bbox2 = output[_b,:,:,10:15]  
                image = None         
                for _i in range(position_class.shape[0]):
                    for _j in range(position_class.shape[1]):
                        for _c in range(5):
                            if position_class[_i][_j][_c] >0.5:
                                if position_xywh_bbox1[_i][_j][4] > position_xywh_bbox2[_i][_j][4]:
                                    bbox_better = position_xywh_bbox1[_i][_j]
                                else:
                                    bbox_better = position_xywh_bbox2[_i][_j]
                                
                                # print(bbox_better[4])
                                if bbox_better[4] > 0.2:
                                    bbox_better = bbox_better[:4].cpu().numpy().flatten()
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

                if image is not None and is_plot:
                    plt.imshow(image)
                    plt.show()
                        


import json
result = {
    "loss": total_loss,
    "accuracy": total_acc
}
with open('results/evaluation_yolov3.json', 'w') as f:
    json.dump(result, f)

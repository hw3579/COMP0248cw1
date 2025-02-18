import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import TotalModel, segmentation_loss
from dataloader import Comp0249Dataset
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
test_dataset = Comp0249Dataset('data/CamVid', "val", transform=None, target_transform=None)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)

is_plot = True

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        output = model(images)
        loss = segmentation_loss(output, labels)
        total_loss.append(loss.item())
        pred_labels = torch.argmax(output, dim=1)
        batch_acc = (pred_labels == labels).float().mean()
        total_acc.append(batch_acc.item())
        if is_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 4, 1)
            show_image = images[0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            labels = labels.cpu().numpy()
            pred_labels = pred_labels.cpu().numpy()
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

import json
result = {
    "loss": total_loss,
    "accuracy": total_acc
}
with open('results/evaluation.json', 'w') as f:
    json.dump(result, f)

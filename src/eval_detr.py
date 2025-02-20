import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from detr import DETR
from dataloader import Comp0249Dataset
from detr import segmentation_loss
import sys
import detr

def evaluate_model(model, dataloader, device, visualize=False):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            # 前向传播得到预测结果
            pred = model((images, labels))
            
            # 计算分割损失（要求 pred['pred_logits'] 的尺寸为 [batch, num_classes, H, W]）
            loss = segmentation_loss(pred, labels)
            total_loss += loss.item()
            
            # 获取每个像素预测的类别
            pred_labels = torch.argmax(pred['pred_logits'], dim=1)
            batch_acc = (pred_labels == labels).float().mean().item()
            total_acc += batch_acc

            # 如果设置了 visualize 参数，则只显示当前批次的第一个样例
            if visualize:
                idx = 0
                # 注意 image 通常需要从 tensor 转换为 numpy，并调整通道顺序
                img = images[idx].cpu().permute(1, 2, 0).numpy()
                gt = labels[idx].cpu().numpy()
                pred_img = pred_labels[idx].cpu().numpy()

                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(img.astype("uint8"))
                plt.title("Input Image")
                plt.subplot(1, 3, 2)
                plt.imshow(gt, cmap='gray')
                plt.title("Ground Truth")
                plt.subplot(1, 3, 3)
                plt.imshow(pred_img, cmap='gray')
                plt.title("Prediction")
                plt.show()
                # 只显示一个批次样例
                # visualize = False

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
    return avg_loss, avg_acc

if __name__ == "__main__":
    sys.modules['__main__'].__dict__.update(detr.__dict__)
    # 假设 Comp0249Dataset 已经定义好，用于加载测试数据
    test_dataset = Comp0249Dataset('data/CamVid', "val", transform=None, target_transform=None)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('results/full_model2.pth', weights_only=False)
    model.to(device)

    # 若有训练好的模型权重，可通过下述方式加载：
    # model.load_state_dict(torch.load("model_weights.pth"))

    # 显示预测图像
    evaluate_model(model, test_loader, device, visualize=True)
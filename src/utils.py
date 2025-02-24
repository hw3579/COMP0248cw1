import cv2 
import numpy as np
# from dataloader import Comp0249Dataset

def draw_the_box(image, pred_label):
    '''
    input: image - the image to draw the box on
            pred_label - the predicted label (RGB image) (H, W, C) PIL image
    output: image - the image with the box drawn on it (H, W)
    '''
    H, W, _ = image.shape
    box_mask = np.zeros((5, H, W), dtype=np.uint8)

    for i in range(5):
        # 得到当前类别的掩码，类型为 bool
        mask = (pred_label == i)
        # 如果该类别没有像素，则跳过
        if np.count_nonzero(mask) == 0:
            continue

        # 获得该类别像素的行和列索引
        rows, cols = np.where(mask)
        top = np.min(rows)
        bottom = np.max(rows)
        left = np.min(cols)
        right = np.max(cols)

        # 保证 image 内存连续
        image = np.ascontiguousarray(image)
        # 绘制矩形框
        cv2.rectangle(image, (left, top), (right, bottom), (0, 25*i, 0), 1)

    return image


import torch
import numpy as np
import cv2

def segmentation_to_yolo(label, S=7, num_classes=20, B=2, scale=1):
    """
    将分割标签 (H, W) 转换为 YOLO 格式标签 (S, S, num_classes + B*5)
    参数：
        label: torch.Tensor, 分割标签，尺寸为 (H, W)，像素值 0 表示背景，其余值对应类别编号（建议连续）
        S: YOLO 网格尺寸（例如 7 表示 7x7）
        num_classes: 类别数
        B: 每个网格预测的边框数，通常这里只用第一个框进行标注，其余保持0
    返回：
        yolo_label: torch.Tensor, 尺寸 (S, S, num_classes + B*5)
                    格式为：前 num_classes 维：目标类别；
                            后面 B*5 维，每个边框组成 [cx, cy, w, h, confidence]
    """
    # 将 label 转为 numpy 数组
    label_np = label.cpu().numpy().astype(np.uint8)
    H, W = label_np.shape
    yolo_label = np.zeros((S, S, B * 5 + num_classes), dtype=np.float32)

    # 使用 OpenCV 连通域分析（假设背景为 0）
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(label_np, connectivity=8)
    stats = np.concatenate([stats, centroids], axis=1)  # 将中心点坐标拼接到 stats 中
    # 过滤掉面积小于 1e3 的区域（背景不计入）
    filtered_stats = stats[stats[:, -3] >= (1e3//scale)].astype(np.int32)
    # 遍历除背景外的每个目标
    for i in range(1, len(filtered_stats)):
        x, y, w, h, area, cx, cy = filtered_stats[i]  # 边框左上角坐标、宽、高、面积       # 边界框中心点 (浮点)
        # 归一化边框参数（相对于整幅图像尺寸）
        norm_cx = cx / W
        norm_cy = cy / H
        norm_w = w / W
        norm_h = h / H

        # 获取目标类别 —— 这里采用区域内出现频率最高的类别编号
        region = label_np[y:y+h, x:x+w]
        # 注意：背景为0，不计入类别统计
        region_flat = region[region > 0]
        if len(region_flat) == 0:
            continue
        class_id = np.bincount(region_flat).argmax()


        # 确定目标中心所在的网格单元
        grid_x = int(norm_cx * S)
        grid_y = int(norm_cy * S)
        grid_x = min(grid_x, S - 1)
        grid_y = min(grid_y, S - 1)

        # 设置该网格单元的类别信息与边框信息（这里只填充第一个边框，后续边框保持0）
        yolo_label[grid_y, grid_x, class_id - 1] = 1
        yolo_label[grid_y, grid_x, num_classes:num_classes+5] = np.array([norm_cx, norm_cy, norm_w, norm_h, 1.0], dtype=np.float32)

    # 转换回 torch.Tensor
    return torch.from_numpy(yolo_label)

def check_binary(matrix: torch.Tensor) -> bool:
    """
    检查输入的矩阵是否只包含 0 和 1

    参数：
        matrix (torch.Tensor): 输入张量

    返回：
        bool: 如果只包含0和1返回True，否则返回False
    """
    return torch.all((matrix == 0) | (matrix == 1)).item()

if __name__ == '__main__':
    test_dataset = Comp0249Dataset('data/CamVid', "train")
    image, label = test_dataset.testgetitem(0)
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
    cv2.imshow('image', image)
    cv2.imshow('label', label*20)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


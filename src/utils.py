import cv2 
import numpy as np
from dataloader import Comp0249Dataset

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


if __name__ == '__main__':
    test_dataset = Comp0249Dataset('data/CamVid', "train")
    image, label = test_dataset.testgetitem(0)
    image = image.numpy().transpose(1, 2, 0).astype(np.uint8)
    label = label.numpy()
    image = draw_the_box(image, label)
    # automatically resize window
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # large the image
    h, w, _ = image.shape
    cv2.resizeWindow('image', 960, 720)
    cv2.imshow('image', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


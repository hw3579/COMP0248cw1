from matplotlib import pyplot as plt
import numpy as np
import json
import math
read_deeplabdata = 0
read_yolov1data = 0
read_yolov3data = 0
read_finaldeeplabdata = 1

if read_deeplabdata:
# 从 JSON 文件中读取数据
    with open('results/deeplabmodeldata.json', 'r') as f:
        data = json.load(f)

    # 提取损失和准确率数据
    loss = data['loss']
    accuracy = data['accuracy']

    # 创建一个新的图形
    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(accuracy, label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()

if read_yolov1data:
    # 从 JSON 文件中读取数据
    with open('results/data.json', 'r') as f:
        data = json.load(f)

    # 提取损失和准确率数据
    loss = data['loss']
    accuracy = data['accuracy']

    # 创建一个新的图形
    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(accuracy, label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()

if read_yolov3data:
    # 从 JSON 文件中读取数据
    with open('results/data_yolov3_optimize3.json', 'r') as f:
        data = json.load(f)

    # 提取损失和准确率数据
    loss = np.array(data['loss'])
    accuracy = data['accuracy']

    # 创建一个新的图形
    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(np.log10(loss), label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(accuracy, label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()

if read_finaldeeplabdata:
    # Load data from first JSON file
    with open('results/deeplabmodeldatafinal_interrupted_20250309_015854.json', 'r') as f:
        data = json.load(f)

    # 提取损失和准确率数据
    loss = np.array(data['loss'])
    accuracy = data['accuracy']


    # 创建一个新的图形
    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(accuracy, label='Accuracy', color='orange')
    # plt.plot(yolo_acc, label='Yolo Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.savefig('results/training_results.png')
    plt.show()
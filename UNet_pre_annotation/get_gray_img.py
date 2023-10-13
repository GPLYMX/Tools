# -*- coding: utf-8 -*-
# @Time : 2023/9/14 17:21
# @Author : GuoPeng
import os
import math
import yaml
import time

from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt

from network import U_Net
from utiles import resize_img


"""
将十三通道的图片放入模型中预测，根据预测的结果生成灰度图，灰度图中的数字代表类别
"""
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('use_GPU:', True)
    device = torch.device('cuda')
else:
    print('use_GPU:', False)
    device = torch.device('cpu')


def load_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


configs = load_configs('configs.yaml')
num_classes = configs['num_classes']
in_channels = configs['in_channels']


def combine_img(folder_path):
    # 获取文件夹中的所有图像文件名
    image_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".png")]
    # 加载灰度图像并添加到列表中
    image_files.sort()
    image_list = []
    for img_path in image_files:
        img = Image.open(os.path.join(folder_path, img_path)).convert("L")  # 将图像转换为灰度模式
        image_list.append(img)

    # 确定图像的尺寸（假设所有图像都有相同的尺寸）
    width, height = image_list[0].size

    # 创建一个空的PyTorch张量，用于存储多通道图像
    multi_channel_image = torch.zeros(len(image_list), height, width)

    # 将灰度图像的像素数据叠加到PyTorch张量中
    for i, img in enumerate(image_list):
        # 将PIL图像转换为PyTorch张量
        img_tensor = transforms.ToTensor()(img)
        # 仅使用灰度通道数据
        multi_channel_image[i] = img_tensor[0]
    return multi_channel_image


def read_img(img_root):
    """
    读取十三通道图片，生成tensor
    :param img_root: 路径应该在通道图片所在文件夹的上一层目录中
    :return:model需要的格式, 原始图片的尺寸
    """
    image = combine_img(img_root)
    h, w = image.shape[1], image.shape[2]
    image = resize_img(image) / 255.

    image = torch.unsqueeze(image, 0)

    return image, [h ,w]


def model_pred(image):
    h = math.ceil(image.shape[2] / 2)
    w = math.ceil(image.shape[3] / 2)
    batch_list = [image[:, :, h:, :w], image[:, :, h:, w:], image[:, :, :h, :w],
                  image[:, :, :h, w:]]

    outputs = []
    model = U_Net(img_ch=in_channels, output_ch=num_classes)
    model.load_state_dict(torch.load('temp.pt'))
    with torch.no_grad():
        for i in range(4):
            model.to(device)
            batch_list[i] = batch_list[i].requires_grad_(False)
            outputs.append(model(batch_list[i].to(device)).to('cpu'))

    outputs = torch.cat([torch.cat([outputs[2], outputs[3]], dim=3), torch.cat([outputs[0], outputs[1]], dim=3)], dim=2)

    return outputs


def get_gray_img(img_root):
    """
    读取十三通道图片，生成灰度图
    :param img_root:
    :return:
    """
    img, shape = read_img(img_root)
    img = model_pred(img)
    img = torch.squeeze(img, 0)
    img = img[:, :shape[0], :shape[1]]

    # img由tensor格式转化为numpy
    try:
        img = img.detach().numpy()
    except:
        img.to('cpu')
        img = img.detach().numpy()

    gray_img = np.argmax(img, axis=0)
    plt.imshow(gray_img)
    plt.show()

    return gray_img


if __name__ == '__main__':
    t1 = time.time()
    root = r'D:\mycodes\RITH\puer\data_20230901_2\data\test\1-test\combined_data'
    img = get_gray_img(root)
    print(time.time() - t1)
    plt.imshow(img)
    plt.show()

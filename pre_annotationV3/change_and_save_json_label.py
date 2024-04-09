# -*- coding: utf-8 -*-
# @Time : 2023/10/26 17:11
# @Author : GuoPeng

import json
import os
import shutil

import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import cv2
import labelme2yolo


def change_json_label(path):
    """
    原始的json文件有五个标签，现在需要把3、4标签转换成2，以方便yolo训练
    :return: 
    """
    # # 1. 解析LabelMe标注文件（JSON格式）
    filename = os.path.basename(path)

    for file in os.listdir(path):
        if file.endswith(".json") and file != str('impurity.json'):
            labelme_json_root = os.path.join(path, file)
            break

    try:
        with open(labelme_json_root, 'r') as json_file:
            labelme_data = json.load(json_file)
    except FileNotFoundError:
        print('没有预标注的json文件')
        return None
    
    # 2. 获取图像尺寸
    image_width = labelme_data['imageWidth']
    image_height = labelme_data['imageHeight']
    for i, shape in enumerate(labelme_data['shapes']):
        if shape['label'] == '2' or shape['label'] == '4':
            labelme_data['shapes'][i]['label'] = '0'
        else:
            del labelme_data['shapes'][i]
    labelme_data = json.dumps(labelme_data)
    f = open(os.path.join(path, 'impurity.json'), 'w')
    f.write(labelme_data)
    f.close()
    return labelme_data


def mkdir(path):
    img_root = os.path.join(path, 'yolo_type', 'images')
    label_root = os.path.join(path, 'yolo_type', 'labels')
    try:
        os.makedirs(os.path.join(path, 'yolo_type'))
    except FileExistsError:
        pass
    try:
        os.makedirs(img_root)
    except FileExistsError:
        pass
    try:
        os.makedirs(label_root)
    except FileExistsError:
        pass

    return img_root, label_root

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
    multi_channel_image = np.zeros((len(image_list), height, width), dtype=np.float32)

    # 将灰度图像的像素数据叠加到PyTorch张量中
    for i, img in enumerate(image_list):
        # 将PIL图像转换为PyTorch张量
        # img_tensor = transforms.ToTensor()(img)
        # 仅使用灰度通道数据
        multi_channel_image[i] = img

    return multi_channel_image


def copyfile(srcfile, dstpath, new_filename):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        # fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, os.path.join(dstpath, new_filename))          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + new_filename))


if __name__ == '__main__':

    data_root = r'D:\mycodes\RITH\puer\datas\data20240229\train'

    img_root, label_root = mkdir(data_root)

    for filename in os.listdir(data_root):

        if filename[2] != 'w':
            file_root = os.path.join(data_root, filename)
            # 生成数据
            change_json_label(file_root)
            # 复制并重命名图片
            origin_img_root = os.path.join(file_root, 'rgb.jpg')
            copyfile(origin_img_root, img_root, filename+'.jpg')
            # 复制并重命名标签
            origin_img_root = os.path.join(file_root, 'impurity.json')
            copyfile(origin_img_root, label_root, filename+'.json')
            # 复制classes.txt
            # origin_img_root = os.path.join(file_root, 'classes.txt')
                # copyfile(origin_img_root, label_root, 'classes.txt')

        # except ValueError:
        #     print(ValueError)
        #     print(filename, "转移失败")
        #     pass

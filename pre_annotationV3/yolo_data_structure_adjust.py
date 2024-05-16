# -*- coding: utf-8 -*-
# @Time : 2023/10/20 10:43
# @Author : GuoPeng

"""
用于调整已有的数据位置，用于训练yolo
"""

import os
import shutil

import cv2
from PIL import Image
import numpy as np

from main import main


def copyfile(srcfile, dstpath, new_filename):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        # fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, os.path.join(dstpath, new_filename))          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + '\\' + new_filename))


def mkdir(path):
    img_root = os.path.join(path, 'initial_check', 'images')
    label_root = os.path.join(path, 'initial_check', 'labels')
    try:
        os.makedirs(os.path.join(path, 'initial_check'))
        os.makedirs(img_root)
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
    for img_path in image_files[:3]:
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

    multi_channel_image = np.transpose(multi_channel_image, (1, 2, 0))
    return multi_channel_image


def adjust():
    pass


if __name__ == '__main__':
    save_root = r'D:\mycodes\RITH\puer\datas\data_dayi\1hao-benxian\data20240510-4'
    data_root = r'D:\mycodes\RITH\puer\datas\data_dayi\1hao-benxian\data20240510-4'

    img_root, label_root = mkdir(save_root)
    # 对文件夹中的每一组数据进行标注
    for filename in os.listdir(data_root):
        temp_path = os.path.join(data_root, filename)
        for i in os.listdir(temp_path):
            if str(i)[0:4] == 'cube' and str(i)[-4] != '.':
                filename = os.path.join(filename, i)
                break

        # int(filename[0])
        # if filename[2] != 'w': # 当不是白板时
        file_root = os.path.join(data_root, filename)
        # 生成数据
        # main(file_root, yolo_flag='no', chaguo_category=2, label3=2)

        # 根据pre_combine的前三个通道生成彩图
        rgb = combine_img(os.path.join(file_root, 'pre_process'))
        cv2.imwrite(os.path.join(img_root, os.path.split(filename)[-1]+'.png'), rgb)

        # # 复制并重命名图片
        # origin_img_root = os.path.join(file_root, 'rgb.png')
        # copyfile(origin_img_root, img_root, filename+'.png')

        # 复制并重命名标签
        origin_img_root = os.path.join(file_root, 'rgb.txt')
        # print('label root：', label_root)
        # print('origin_img_root：', origin_img_root)
        copyfile(origin_img_root, label_root, os.path.split(filename)[-1]+'.txt')
        # 复制classes.txt
        origin_img_root = os.path.join(file_root, 'classes.txt')
        copyfile(origin_img_root, label_root, 'classes.txt')

        # except Exception as e:
        #     print(e)
        #     pass

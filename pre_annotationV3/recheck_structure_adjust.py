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
    img_root = os.path.join(path, 'recheck', 'images')
    label_root = os.path.join(path, 'recheck', 'labels')
    try:
        os.makedirs(os.path.join(path, 'recheck'))
        os.makedirs(img_root)
        os.makedirs(label_root)
    except FileExistsError:
        pass

    return img_root, label_root


def adjust():
    pass


if __name__ == '__main__':
    save_root = r'D:\mycodes\RITH\puer\datas\data_dayi\1hao-benxian\data20240510-4'
    data_root = r'D:\mycodes\RITH\puer\datas\data_dayi\1hao-benxian\data20240510-4'

    img_root, label_root = mkdir(save_root)
    # 对文件夹中的每一组数据进行标注
    for filename in os.listdir(data_root):
        new_name = ''
        temp_path = os.path.join(data_root, filename)
        if 'cube' not in filename:
            for i in os.listdir(temp_path):
                if str(i)[0:4] == 'rech' and str(i)[-4] != '.':
                    filename = os.path.join(filename, i)
                if str(i)[0:4] == 'cube' and str(i)[-4] != '.':
                    new_name = i
        else:
            new_name = filename
            for i in os.listdir(temp_path):
                if str(i)[0:4] == 'rech' and str(i)[-4] != '.':
                    filename = os.path.join(filename, i)

        # int(filename[0])
        # if filename[2] != 'w': # 当不是白板时
        file_root = os.path.join(data_root, filename)
        # 复制并重命名图片
        origin_img_root = os.path.join(file_root, 'processed_photo.png')
        print(origin_img_root)
        print(img_root)
        copyfile(origin_img_root, img_root, new_name+'.png')

        # # 复制并重命名标签
        # origin_img_root = os.path.join(file_root, 'rgb.txt')
        # # print('label root：', label_root)
        # # print('origin_img_root：', origin_img_root)
        # copyfile(origin_img_root, label_root, os.path.split(filename)[-1]+'.txt')
        # # 复制classes.txt
        # origin_img_root = os.path.join(file_root, 'classes.txt')
        # copyfile(origin_img_root, label_root, 'classes.txt')

        # except Exception as e:
        #     print(e)
        #     pass

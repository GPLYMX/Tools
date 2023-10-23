# -*- coding: utf-8 -*-
# @Time : 2023/10/20 10:43
# @Author : GuoPeng

"""
用于调整已有的数据位置，用于训练yolo
"""

import os
import shutil

from main import main


def copyfile(srcfile, dstpath, new_filename):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        # fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, os.path.join(dstpath, new_filename))          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + new_filename))


def mkdir(path):
    img_root = os.path.join(path, 'yolo_type', 'images')
    label_root = os.path.join(path, 'yolo_type', 'labels')
    try:
        os.makedirs(os.path.join(path, 'yolo_type'))
        os.makedirs(img_root)
        os.makedirs(label_root)
    except FileExistsError:
        pass

    return img_root, label_root


def adjust():
    pass


if __name__ == '__main__':
    save_root = r'D:\mycodes\RITH\puer\data_20231020\train'
    data_root = r'D:\mycodes\RITH\puer\data_20231020\train'

    img_root, label_root = mkdir(save_root)
    # 对文件夹中的每一组数据进行标注
    for filename in os.listdir(data_root):
        try:
            int(filename[0])
            if filename[2] != 'w':
                file_root = os.path.join(data_root, filename)
                # 生成数据
                main(file_root)

                # 复制并重命名图片
                origin_img_root = os.path.join(file_root, 'rgb.jpg')
                copyfile(origin_img_root, img_root, filename+'.jpg')
                # 复制并重命名标签
                origin_img_root = os.path.join(file_root, 'rgb.txt')
                copyfile(origin_img_root, label_root, filename+'.txt')
                # 复制classes.txt
                origin_img_root = os.path.join(file_root, 'classes.txt')
                copyfile(origin_img_root, label_root, 'classes.txt')

        except ValueError:
            pass

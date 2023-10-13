# -*- coding: utf-8 -*-
# @Time : 2023/9/15 14:21
# @Author : GuoPeng
import yaml
import math

import numpy as np
import cv2

"""
将灰度图转化为json格式
"""


def load_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


configs = load_configs('configs.yaml')


def add_shapes(labelme_data, temp_mask, offset_h, offset_w,  label_name):

    contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:

        points = contour.squeeze().tolist()
        print(points)
        print(len(points))
        if len(points) <= 20:
            continue
        if isinstance(points[0], list):
            # 修改points的偏移量
            for i in range(len(points)):
                points[i][0] += offset_w
                points[i][1] += offset_h
            shape_data = {
                "label": label_name,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
            labelme_data["shapes"].append(shape_data)
    return labelme_data


def gray_to_labelme(gray_image, class_mapping=configs['class_mapping']):
    """
    读取灰度图，转化为json文件
    :param gray_image:np.numpy格式的灰度图
    :param class_mapping:
    :return:labelme可读取的json文件
    """
    height, width = gray_image.shape[0], gray_image.shape[1]
    base_h, base_w = configs['base_size']

    labelme_data = {
        "version": "5.3.1",
        "flags": {},
        "shapes": [],
        "imagePath": '00.png',  # 请替换成实际的图像文件名
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    for label_value, label_name in class_mapping.items():
        if label_name == "background" or label_name == "tea":
            continue
        if label_name == "impurity":
            base_h, base_w = 5*base_h, 5*base_w

        iter_num_h, iter_num_w = math.ceil(height / base_h), math.ceil(width / base_w)
        boundary_pixel_num_h, boundary_pixel_num_w = height % base_h, width % base_w
        padding_pixel_num_h, padding_pixel_num_w = (base_h - boundary_pixel_num_h) % base_h, (
                base_w - boundary_pixel_num_w) % base_w

        mask = (gray_image == label_value).astype(np.uint8)
        mask = np.pad(mask, ((0, padding_pixel_num_h), (0, padding_pixel_num_w)), 'constant', constant_values=(0, 0))

        for i in range(iter_num_h):
            for j in range(iter_num_w):
                offset_h, offset_w = i * base_h, j * base_w
                temp_mask = mask[offset_h:base_h + offset_h, offset_w:base_w + offset_w]

                labelme_data = add_shapes(labelme_data, temp_mask, offset_h, offset_w, label_name)

    return labelme_data

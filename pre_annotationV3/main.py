# -*- coding: utf-8 -*-
# @Time : 2023/10/20 9:15
# @Author : GuoPeng

"""
代码功能：本代码用于生成训练unet所需要的灰度图、训练yolo需要的txt文件
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import cv2


def thredshold_seg(img_root, thredshold=70):
    """
    输入0通道的灰度图，然后返回二值灰度图
    :param img_root:
    :return:
    """
    img_name_list = ['01.png', '02.png', '03.png', ]
    img_list = []
    for i, m in enumerate(img_name_list):
        img_list.append(cv2.imread(os.path.join(img_root, m), cv2.IMREAD_GRAYSCALE))
    # img = cv2.imread(img_root,cv2.IMREAD_GRAYSCALE)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img_list)
    img = img.transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[img >= thredshold] = 255
    img[img < thredshold] = 1
    img[img == 255] = 0
    plt.imshow(img)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.axis('off')
    plt.title('阈值分割标注结果')
    plt.show()
    return img


def rectangle_to_yolo(rectangle, image_width, image_height):
    min_x, min_y, max_x, max_y = rectangle

    # 计算中心坐标
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # 计算宽度和高度
    width = max_x - min_x
    height = max_y - min_y

    # 归一化坐标和尺寸到图像尺寸范围内
    center_x = center_x / image_width
    center_y = center_y / image_height
    width = width / image_width
    height = height / image_height

    return f"0 {center_x} {center_y} {width} {height}"


def labelme_to_gray_yolo(path, chaguo_category=2):
    """
    读取labelme生成的json文件，然后转化成灰度图，灰度图中的数值代表像素点所属的类别
    """
    # # 1. 解析LabelMe标注文件（JSON格式）
    filename = os.path.basename(path)
    labelme_json_root = os.path.join(path, 'RGB.json')
    try:
        with open(labelme_json_root, 'r') as json_file:
            labelme_data = json.load(json_file)
    except FileNotFoundError:
        print('没有预标注的json文件')
        return None

    # 2. 获取图像尺寸
    image_width = labelme_data['imageWidth']
    image_height = labelme_data['imageHeight']

    # 3. 创建灰度图
    gray_image = Image.new('L', (image_width, image_height), 0)

    # 4. 为每个对象分配类别值
    category_mapping = {}  # 用于将类别名称映射到整数值
    category_id = 1

    # 5. 创建保存yolo信息的列表
    yolo_data = []

    for shape in labelme_data['shapes']:
        category_name = shape['label']
        if category_name not in category_mapping:
            category_mapping[category_name] = category_id
            category_id += 1

        category_value = 0
        if shape['label'] == '1' or shape['label'] == 'tea':
            category_value = 1
        if shape['label'] == '2' or shape['label'] == '3' or shape['label'] == 'impurity' or shape['label'] == 'impurities':
            category_value = 2
        if shape['label'] == '4':
            category_value = chaguo_category
        if isinstance(shape['points'][0], list):
            # 创建多边形的坐标列表
            polygon_points = [(int(x), int(y)) for x, y in shape['points']]

            # 使用PIL的绘图功能填充多边形区域
            draw = ImageDraw.Draw(gray_image)
            draw.polygon(polygon_points, fill=category_value)

        # # 获取yolo格式的坐标
        # if category_name == '4':
        #     # 获取该形状的位置信息
        #     points = shape['points']
        #     # 计算最小矩形区域
        #     x_coordinates, y_coordinates = zip(*points)
        #     min_x = min(x_coordinates)
        #     max_x = max(x_coordinates)
        #     min_y = min(y_coordinates)
        #     max_y = max(y_coordinates)
        #     rectangle = min_x, min_y, max_x, max_y
        #     yolo_data.append(rectangle_to_yolo(rectangle, image_width, image_height))
        # 
        #     # 绘制茶果矩形框
        #     draw = ImageDraw.Draw(gray_image)
        #     draw.rectangle(xy=[(min_x, min_y), (max_x, max_y)], outline=2, fill=0, width=5)

    # 将YOLO格式数据保存到txt文件
    with open(os.path.join(path, 'RGB.txt'), 'w') as txt_file:
        for line in yolo_data:
            txt_file.write(line + '\n')
    with open(os.path.join(path, 'classes.txt'), 'w') as txt_file:
        txt_file.write('0')

    image = np.array(gray_image)
    gray_image = Image.fromarray(image)
    gray_image = gray_image.convert('L')
    # gray_image.save('output_gray_image.png')
    plt.imshow(gray_image)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.axis('off')
    plt.title(filename+'json标注结果')
    plt.show()
    return image


def main(path):
    # 1、先使用阈值分割，将茶叶和杂质视为一类、背景视为一类。得到二值灰度图：像素0代表背景、像素1代表茶叶和杂质
    seg_label = thredshold_seg(os.path.join(path, 'pre_process'))
    # 2、使用lableme手动标注的json文件，获取杂质标签，生成灰度图：0代表背景、1代表茶叶、2代表杂质。
    json_label = labelme_to_gray_yolo(path)

    # 3、如果存在json文件、合并两个灰度图
    if json_label.any():
        seg_label[json_label == 2] = 2

    # 4、保存灰度图:灰度图名称为label.png，保存在pretreatment文件夹下
    im = Image.fromarray(seg_label)
    im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
    im.save(os.path.join(path, 'label.png'))
    print('标签图label.png保存成功')

    # 5、保存可视化的掩码图，名称为visual.png，保存在pretreatment文件夹下
    seg_label = seg_label * 100
    im = Image.fromarray(seg_label)
    im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
    im.save(os.path.join(path, 'visual.png'))
    print('可视化图visual.png保存成功')

    return seg_label


if __name__ == "__main__":
    """
    一、数据格式：参见测试用例数据1-mix2_20231018_102549
    二、生成以下文件，并保存在测试用例文件夹下：
         1、rgb.txt，这是yolo格式的标签，用于检测茶果
         2、label.png，这是用于训练unet的标签，格式是灰度图，灰度值代表所属类别：0、背景；1、茶叶、2杂质。(茶果类别可以在函数labelme_to_gray_yolo中修改参数chaguo_category)
         3、visual.png，这是对label.png的可视化，可以直接点击查看标注的效果。其中：茶叶、杂质是像素级标注；茶果用方框标记
         4、classes.txt，当需要用labelimg调整rgb.txt文件时，需要该文件。此外没有其它意义
    """

    path = r'D:\mycodes\RITH\puer\data_20231018\train\2-mix2_20231018_111434'
    gray_img = main(path)

    # 重新读取并显示灰度图，验证是否正确
    root = os.path.join(path, 'label.png')
    img = cv2.imread(root, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('读取保存好的灰度图')
    plt.axis('off')
    plt.show()

# -*- coding: utf-8 -*-
# @Time : 2023/9/21 17:40
# @Author : GuoPeng
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import cv2


def thredshold_seg(img_root, thredshold=150):
    """
    输入0通道的灰度图，然后返回二值灰度图
    :param img_root:
    :return:
    """
    img_name_list = ['00.png', '01.png', '02.png']
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


def labelme_to_gray(labelme_json_root):
    """
    读取labelme生成的json文件，然后转化成灰度图，灰度图中的数值代表像素点所属的类别
    """
    # # 1. 解析LabelMe标注文件（JSON格式）
    labelme_json_root = os.path.join(labelme_json_root, 'pretreatment', 'RGB.json')
    try:
        with open(labelme_json_root, 'r') as json_file:
            labelme_data = json.load(json_file)
    except FileNotFoundError:
        print('没有json文件')
        return None

    # 2. 获取图像尺寸
    image_width = labelme_data['imageWidth']
    image_height = labelme_data['imageHeight']

    # 3. 创建灰度图
    gray_image = Image.new('L', (image_width, image_height), 0)

    # 4. 为每个对象分配类别值
    category_mapping = {}  # 用于将类别名称映射到整数值
    category_id = 1

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
        if isinstance(shape['points'][0], list):
            # 创建多边形的坐标列表
            polygon_points = [(int(x), int(y)) for x, y in shape['points']]

            # 使用PIL的绘图功能填充多边形区域
            draw = ImageDraw.Draw(gray_image)
            draw.polygon(polygon_points, fill=category_value)

    image = np.array(gray_image)
    gray_image = Image.fromarray(image)
    gray_image = gray_image.convert('L')
    # gray_image.save('output_gray_image.png')
    plt.imshow(gray_image)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.axis('off')
    plt.title('json标注结果')
    plt.show()
    return image


def main(path):
    # 1、先使用阈值分割，将茶叶和杂质视为一类、背景视为一类。得到二值灰度图：像素0代表背景、像素1代表茶叶和杂质
    seg_label = thredshold_seg(os.path.join(path, 'combined_data'))
    # 2、使用lableme手动标注的json文件，获取杂质标签，生成灰度图：0代表背景、1代表茶叶、2代表杂质。json放置在pretreatment文件夹下，名称为00.json
    json_label = labelme_to_gray(path)

    # 3、如果存在json文件、合并两个灰度图
    if json_label.any():
        seg_label[json_label == 2] = 2

    # 4、保存灰度图:灰度图名称为label.png，保存在pretreatment文件夹下
    im = Image.fromarray(seg_label)
    im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
    im.save(os.path.join(path, 'pretreatment', 'label.png'))
    print('灰度图保存成功')

    # 5、保存可视化的掩码图，名称为visual.png，保存在pretreatment文件夹下
    seg_label = seg_label * 100
    im = Image.fromarray(seg_label)
    im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
    im.save(os.path.join(path, 'pretreatment', 'visual.png'))
    print('灰度图保存成功')



    return seg_label


if __name__ == "__main__":
    """
    文件位置须知：
    1、需要把labelme标注的RGB.json文件放在pretreatment文件夹里面
    2、生成的标签用灰度图来表示，名称为label.png，放置在pretreatment文件夹下
    """
    path = r'D:\mycodes\RITH\puer\data_20230921\mixed_samples\1-mix2_20230921_103146'
    gray_img = main(path)

    # 重新读取并显示灰度图，验证是否正确
    root = os.path.join(path, 'label.png')
    img = cv2.imread(root, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('读取保存好的灰度图')
    plt.axis('off')
    plt.show()

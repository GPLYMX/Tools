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
import labelme2yolo


def thredshold_seg(img_root, thredshold=70):
    """
    输入0通道的灰度图，然后返回二值灰度图
    :param img_root:
    :return:
    """
    img_name_list = ['00.png', '01.png', '02.png', ]
    img_list = []
    for i, m in enumerate(img_name_list):
        img_list.append(cv2.imread(os.path.join(img_root, m), cv2.IMREAD_GRAYSCALE))
    a = cv2.imread(os.path.join(img_root, m), cv2.IMREAD_GRAYSCALE)

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


def rectangle_to_yolo(label=4, rectangle=0, image_width=0, image_height=0):
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

    return f"{int(label)-4} {center_x} {center_y} {width} {height}"


def labelme_to_gray_yolo(path, yolo_flag='patial', chaguo_category=2, label3=1):
    """
    读取labelme生成的json文件，然后转化成灰度图，灰度图中的数值代表像素点所属的类别
    :param path: 数据路径
    :param yolo_flag: 代表是否生成yolo标签，‘no’代表不生成yolo标签;'patial'代表只生成茶果的yolo标签;‘all’代表所有杂质都生成yolo标签
    :param chaguo_category: 定义茶果所属类别
    :param label3: 茶叶覆盖茶果时，标签为多少
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

    if yolo_flag == 'all':
        # 如果所有的杂质都使用yolo格式，那么茶果应该是2类别，重叠部分也应该是2类别
        chaguo_category = 2
        label3 = 2

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
        if shape['label'] == '3':
            if yolo_flag == 'all':
                category_value = 2
            else:
                category_value = label3
        if shape['label'] == '1' or shape['label'] == 'tea':
            category_value = 1
        if shape['label'] == '2':
            category_value = 2
        if shape['label'] == '3':
            category_value = 1
        if shape['label'] == '4':
            category_value = 3
        if shape['label'] == '5':
            category_value = 4
        if shape['label'] == '6':
            category_value = 3
        if isinstance(shape['points'][0], list):
            # 创建多边形的坐标列表
            polygon_points = [(int(x), int(y)) for x, y in shape['points']]
            # 使用PIL的绘图功能填充多边形区域
            draw = ImageDraw.Draw(gray_image)
            draw.polygon(polygon_points, fill=category_value)

        # 获取yolo格式的坐标
        if yolo_flag == 'patial':
            if category_name == '4' or category_name == '5':
                # 获取该形状的位置信息
                points = shape['points']
                # 计算最小矩形区域
                x_coordinates, y_coordinates = zip(*points)
                min_x = min(x_coordinates)
                max_x = max(x_coordinates)
                min_y = min(y_coordinates)
                max_y = max(y_coordinates)
                rectangle = min_x, min_y, max_x, max_y
                yolo_data.append(rectangle_to_yolo(category_name, rectangle, image_width, image_height))

                # 绘制茶果矩形框
                draw = ImageDraw.Draw(gray_image)
                draw.rectangle(xy=[(min_x, min_y), (max_x, max_y)], outline=2, fill=0, width=5)

    if yolo_flag == 'all':
        yolo_img = gray_image.copy()
        yolo_img = np.array(yolo_img)
        yolo_img[yolo_img==1] = 0
        yolo_img[yolo_img==2] = 1
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(yolo_img)
        for label in range(1, len(stats)):
            pixels_in_component = np.argwhere(labels == label)
            y_coordinates, x_coordinates = zip(*pixels_in_component)
            min_x = min(x_coordinates)
            max_x = max(x_coordinates)
            min_y = min(y_coordinates)
            max_y = max(y_coordinates)
            rectangle = min_x, min_y, max_x, max_y
            yolo_data.append(rectangle_to_yolo(rectangle, image_width, image_height))

            # 绘制杂质矩形框
            draw = ImageDraw.Draw(gray_image)
            draw.rectangle(xy=[(min_x, min_y), (max_x, max_y)], outline=2, fill=0, width=5)

    # 将YOLO格式数据保存到txt文件
    if yolo_flag != 'no':
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


def main(path, yolo_flag='all', chaguo_category=2, label3=1):
    """

    :param path: 数据路径
    :param yolo_flag: 代表是否生成yolo标签，‘no’代表不生成yolo标签;'patial'代表只生成茶果的yolo标签;‘all’代表所有杂质都生成yolo标签
    :param chaguo_category: 定义茶果所属类别
    :param label3: 茶叶覆盖茶果时，标签为多少
    :return:
    """
    # 1、先使用阈值分割，将茶叶和杂质视为一类、背景视为一类。得到二值灰度图：像素0代表背景、像素1代表茶叶和杂质
    seg_label = thredshold_seg(os.path.join(path, 'pre_process'))
    # 2、使用lableme手动标注的json文件，获取杂质标签，生成灰度图：0代表背景、1代表茶叶、2代表杂质。
    json_label = labelme_to_gray_yolo(path, yolo_flag=yolo_flag, chaguo_category=chaguo_category, label3=label3)

    # 3、如果存在json文件、合并两个灰度图
    if json_label.any():
        seg_label[json_label == 2] = 2
        seg_label[json_label == 3] = 3
        seg_label[json_label == 4] = 4
        seg_label[json_label == 5] = 5

    # 4、保存灰度图:灰度图名称为label.png，保存在pretreatment文件夹下
    print(seg_label.max())
    im = Image.fromarray(seg_label)
    im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
    im.save(os.path.join(path, 'label.png'))
    print('标签图label.png保存成功')

    # 5、保存可视化的掩码图，名称为visual.png，保存在pretreatment文件夹下
    seg_label = seg_label * 50
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
    三、参数配置：
         详见main函数注释
    """
    dir = r'D:\mycodes\RITH\puer\datas\data20240229\test'
    for pp in os.listdir(dir):
        path = os.path.join(dir, pp)
        print('成功：', path)
        gray_img = main(path, yolo_flag='no', chaguo_category=3, label3=1)

        # 重新读取并显示灰度图，验证是否正确
        root = os.path.join(path, 'label.png')
        img = cv2.imread(root, cv2.IMREAD_GRAYSCALE)
        plt.imshow(img)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.title(pp+'读取保存好的灰度图')
        plt.axis('off')
        plt.show()
        # except Exception as e:
        #     path = os.path.join(root, pp)
        #     print('失败：', path)
        #     print(e)

    # path = r'D:\mycodes\RITH\puer\data20231211\data20231211\cube_20231211_101510'
    # gray_img = main(path, yolo_flag='no', chaguo_category=2, label3=2)
    #
    # # 重新读取并显示灰度图，验证是否正确
    # root = os.path.join(path, 'label.png')
    # img = cv2.imread(root, cv2.IMREAD_GRAYSCALE)
    # plt.imshow(img)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.title('读取保存好的灰度图')
    # plt.axis('off')
    # plt.show()
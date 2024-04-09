# -*- coding: utf-8 -*-
# @Time : 2024/3/18 10:32
# @Author : GuoPeng

import os
import json


def convert_labelme_to_yolo(json_file, class_map):
    with open(json_file, 'r') as f:
        data = json.load(f)

    img_name = os.path.splitext(os.path.basename(json_file))[0]
    txt_file = os.path.join(os.path.dirname(json_file), img_name + '.txt')

    img_width = data['imageWidth']
    img_height = data['imageHeight']

    with open(txt_file, 'w') as f:
        for shape in data['shapes']:
            label = shape['label']
            # Ensure label is present in the class map
            if label not in class_map:
                print(f"Warning: Label '{label}' not found in class map.")
                continue
            class_id = class_map[label]
            # points = shape['points']
            # x_min = min(point[0] for point in points)
            # y_min = min(point[1] for point in points)
            # x_max = max(point[0] for point in points)
            # y_max = max(point[1] for point in points)
            #
            # # Convert to YOLO format
            # x_center = (x_min + x_max) / (2 * img_width)
            # y_center = (y_min + y_max) / (2 * img_height)
            # width = (x_max - x_min) / img_width
            # height = (y_max - y_min) / img_height
            points = shape['points']
            # 计算最小矩形区域
            x_coordinates, y_coordinates = zip(*points)
            min_x = min(x_coordinates)
            max_x = max(x_coordinates)
            min_y = min(y_coordinates)
            max_y = max(y_coordinates)
            rectangle = min_x, min_y, max_x, max_y
            x_center, y_center, width, height = rectangle_to_yolo(rectangle, img_width, img_height)
            print(class_id, x_center, y_center, width, height )

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def rectangle_to_yolo(rectangle=0, image_width=0, image_height=0):
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

    return center_x, center_y, width, height

def main(folder, class_map={'shuzhi':'0', 'bianzhidai':'1', 'qita':'0'}):
    try:
        for file in os.listdir(folder):
            if file.endswith('.json'):
                json_file = os.path.join(folder, file)
                try:
                    convert_labelme_to_yolo(json_file, class_map)
                    print(f"Converted {json_file} successfully.")
                except Exception as e:
                    print(f"Error converting {json_file}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    folder_path = r'D:\mycodes\RITH\puer\datas\recheck_data_20240327\original_data'
    main(folder_path)

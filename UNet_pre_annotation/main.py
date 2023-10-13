# -*- coding: utf-8 -*-
# @Time : 2023/9/15 14:21
# @Author : GuoPeng
import json
import os

from get_gray_img import get_gray_img
from gray_to_labelme import gray_to_labelme


def main(root):
    gray_img = get_gray_img(root)
    json_file = gray_to_labelme(gray_img)

    json_root = os.path.join(root, '00.json')
    with open(json_root, "w") as f:
        f.write(json.dumps(json_file, ensure_ascii=False, indent=4, separators=(',', ':')))
        print('预标注文件生成成功')


if __name__ == "__main__":
    path = r'D:\mycodes\RITH\puer\data_20230901_2\data\test\1-test\combined_data'
    main(path)

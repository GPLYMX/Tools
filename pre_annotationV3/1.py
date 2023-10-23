# -*- coding: utf-8 -*-
# @Time : 2023/10/21 16:22
# @Author : GuoPeng

# 重新读取并显示灰度图，验证是否正确
import cv2
import matplotlib.pyplot as plt
root = r'D:\mycodes\RITH\puer\data_20231018\train\1-mix1_20231018_101004\label.png'
img = cv2.imread(root, cv2.IMREAD_GRAYSCALE)
plt.imshow(img)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title('读取保存好的灰度图')
plt.axis('off')
plt.show()
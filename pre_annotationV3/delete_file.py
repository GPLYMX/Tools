# -*- coding: utf-8 -*-
# @Time : 2024/4/30 10:49
# @Author : GuoPeng

import os
import shutil

def delete_files_in_folder(folder_path, subfolders=[]):
    """
    删除文件夹中的文件以及指定子文件夹中的文件。

    参数：
    folder_path (str): 要操作的文件夹路径。
    subfolders (list): 要删除文件的子文件夹列表。

    返回：
    无返回值。

    """
    # 删除指定文件夹中的文件
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.exists(subfolder_path):
            try:
                shutil.rmtree(subfolder_path)
                print(f"已删除文件夹 {subfolder_path} 及其所有内容。")
            except Exception as e:
                shutil.os.remove(subfolder_path)
                print(f"已删除文件 {subfolder_path}")
            except Exception as e:
                print(f"无法删除文件夹 {subfolder_path}，错误信息：{e}")
        else:
            print(f"文件夹 {subfolder_path} 不存在，跳过删除操作。")

def delete_single_file(folder_path, suffixes=[]):
    """删除后缀名在subfolders的文件"""
    for file in os.listdir(folder_path):
        if os.path.splitext(file)[-1] in suffixes:
            shutil.os.remove(os.path.join(folder_path, file))
            print(f'已删除文件{os.path.join(folder_path, file)}')

if __name__ == "__main__":

    # dir = r'D:\mycodes\RITH\puer\datas\data_dayi\1hao-benxian\data20240510-4\temp\val'
    # for pp in os.listdir(dir):
    #     path = os.path.join(dir, pp)
    #     for i in os.listdir(path):
    #         if str(i)[0:4] == 'cube' and str(i)[-4] != '.':
    #             path = os.path.join(path, i)
    #         if str(i)[0:4] == 'rech' and str(i)[-4] != '.':
    #             path2 = os.path.join(path, i)
    #             shutil.rmtree(path2)
    #     delete_files_in_folder(path, subfolders=['impurities', "pos_0", "pos_1", "pos_2", "pos_3", 'rgb.json', 'rgb.png'])

    delete_single_file((r'D:\mycodes\RITH\puer\datas\data_dayi\1hao-benxian\data20240512\initial_check\images'), suffixes=['.txt'])


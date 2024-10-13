#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：cls_template 
@File    ：data_split.py
@Author  ：ChenmingSong
@Date    ：2022/1/9 19:43 
@Description：
'''
# -*- coding: utf-8 -*-
# @Time    : 2021/6/17 20:29
# @Author  ：dejahu
# @Email   ：1148392984@qq.com
# @File    ：data_split.py
# @Software：PyCharm
# @Brief   ：将数据集划分为训练集、验证集和测试集
import os
import random
import shutil
from shutil import copy2
import os.path as osp

def data_set_split(src_data_folder, target_data_folder, train_scale=0.7, val_scale=0.2, test_scale=0.1):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/src_data
    :param target_data_folder: 目标文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/target_data
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    '''
    print("开始数据集划分")
    class_names = ['hologram', 'reconstruction']
    # 在目标目录下创建文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        # 然后在split_path的目录下创建类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    # 获取所有图像对
    holo_folder = os.path.join(src_data_folder, class_names[0])
    re_folder = os.path.join(src_data_folder, class_names[1])
    holo_images = sorted(os.listdir(holo_folder))
    re_images = sorted(os.listdir(re_folder))

    # 确保图像对数量一致
    assert len(holo_images) == len(re_images), "holo 和 re 文件夹中的图像数量不一致"

    # 打乱图像对
    paired_images = list(zip(holo_images, re_images))
    random.shuffle(paired_images)

    # 按照比例划分数据集
    total_length = len(paired_images)
    train_stop_flag = int(total_length * train_scale)
    val_stop_flag = int(total_length * (train_scale + val_scale))

    train_num = 0
    val_num = 0
    test_num = 0

    for idx, (holo_img, re_img) in enumerate(paired_images):
        if idx < train_stop_flag:
            split_folder = 'train'
            train_num += 1
        elif idx < val_stop_flag:
            split_folder = 'val'
            val_num += 1
        else:
            split_folder = 'test'
            test_num += 1

        # 复制图像对到相应的文件夹
        src_holo_path = os.path.join(holo_folder, holo_img)
        src_re_path = os.path.join(re_folder, re_img)
        dst_holo_path = os.path.join(target_data_folder, split_folder, class_names[0], holo_img)
        dst_re_path = os.path.join(target_data_folder, split_folder, class_names[1], re_img)
        copy2(src_holo_path, dst_holo_path)
        copy2(src_re_path, dst_re_path)

    print("*********************************数据集划分完成*************************************")
    print("数据集按照{}：{}：{}的比例划分完成，一共{}对图像".format(train_scale, val_scale, test_scale, total_length))
    print("训练集：{}对".format(train_num))
    print("验证集：{}对".format(val_num))
    print("测试集：{}对".format(test_num))


if __name__ == '__main__':
    csv = r'M:\Data\AutoFocusDatabase\AutoFocusDatabase.csv'
    src_data_folder = r"M:\Data\AutoFocusDatabase0"  # todo 修改你的原始数据集路径
    target_data_folder = src_data_folder + "_" + "split"
    if osp.isdir(target_data_folder):
        print("target folder 已存在， 正在删除...")
        shutil.rmtree(target_data_folder)
    os.mkdir(target_data_folder)
    print("Target folder 创建成功")

    data_set_split(src_data_folder, target_data_folder)
    print("*****************************************************************")
    print("数据集划分完成，请在{}目录下查看".format(target_data_folder))
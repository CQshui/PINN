# -*- coding: utf-8 -*-
"""
@ Time:     2024/10/14 19:59 2024
@ Author:   CQshui
$ File:     utils.py
$ Software: Pycharm
"""
# 导入必备的包
import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import math
# 网络模型构建需要的包
import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, cross_val_score
# Metric 测试准确率需要的包
from sklearn.metrics import f1_score, accuracy_score, recall_score
# Augmentation 数据增强要使用到的包
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import datasets, models, transforms


# def accuracy(output, target):
#     y_pred = torch.softmax(output, dim=1)
#     y_pred = torch.argmax(y_pred, dim=1).cpu()
#     target = target.cpu()
#
#     return accuracy_score(target, y_pred)


def accuracy(z_pred, z_true, reconstruction_pred, reconstruction_true, hologram, reconstruction_complex):  # 计算准确率
    ''' 移动张量到cpu，减轻gpu负荷 '''
    z_pred = z_pred.detach().cpu().numpy()
    z_true = z_true.detach().cpu().numpy()
    reconstruction_pred = reconstruction_pred.detach().cpu().numpy()
    reconstruction_true = reconstruction_true.detach().cpu().numpy()
    hologram = hologram.detach().cpu().numpy()
    reconstruction_complex = reconstruction_complex.detach().cpu().numpy()

    # 计算预测值与真实值之间的差异。注意：z_true是做了放缩的
    z_accuracy = accuracy_score(z_pred.flatten(), z_true.flatten())
    reconstruction_accuracy = accuracy_score(reconstruction_pred.flatten(), reconstruction_true.flatten())
    hologram_accuracy = accuracy_score(hologram.flatten(), reconstruction_complex.flatten())

    return z_accuracy, reconstruction_accuracy, hologram_accuracy


def show_image(tensor):
    if tensor.size(1) == 2:
        # 针对torch.Size([4, 2, 1, 128, 128])形式的张量
        # 提取第一个样本的第一个通道
        sample = tensor[0, :, 0, :, :]  # 形状为 [2, 128, 128]

        # 提取实部和虚部
        real_part = sample[0, :, :]  # 形状为 [128, 128]
        imag_part = sample[1, :, :]  # 形状为 [128, 128]

        # 计算平方和的平方根
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)  # 形状为 [128, 128]

        # 显示图像
        plt.imshow(magnitude.cpu().numpy(), cmap='gray')
        plt.title('Hologram Image')
        plt.show(block=True)

    elif tensor.size(1) == 1:
        # 针对torch.Size([4, 1, 1, 128, 128])形式的张量
        # 提取第一个样本的第一个通道
        sample = tensor[0, 0, 0, :, :]  # 形状为 [128, 128]

        # 显示图像
        plt.imshow(sample.cpu().numpy(), cmap='gray')
        plt.title('Reconstruction Image')
        plt.show(block=True)

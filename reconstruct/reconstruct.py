# -*- coding: utf-8 -*-
"""
@ Time:     2024/10/12 21:26 2024
@ Author:   CQshui
$ File:     reconstruct.py
$ Software: Pycharm
"""
import cv2
import numpy as np
from FFTshift import fft_filter
import csv
from Fresnel_co_Batch import fresnel
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


csv_path = r'M:\Data\AutoFocusDatabase\AutoFocusDatabase.csv'
hologram_path = r'M:\Data\AutoFocusDatabase\hologram'
result_path = r'M:\Data\AutoFocusDatabase\result'

with open(csv_path, 'r') as f:
    reader = csv.reader(f)

    # 跳过第一行
    next(reader)

    # 获取总行数
    total_rows = sum(1 for _ in reader)
    f.seek(0)  # 重置文件指针
    next(reader)  # 再次跳过第一行

    # row: ['532', '0.098', '0.0003600', 'Image__2024-04-26__20-23-57.bmp', 'Image__2024-04-26__20-23-57_0.0003600.jpg']
    for row in tqdm(reader, total=total_rows, desc="Processing"):
        holo_img = cv2.imread(os.path.join(hologram_path, row[3]), 0)
        holo_filtered = fft_filter(holo_img, x=2152, y=2164, w=212, h=124)
        reconstruction = fresnel(holo_filtered, lam=532, pix=0.098, z1=float(row[2]))
        # print(row[2])

        plt.imsave(result_path+'/'+row[4], reconstruction, cmap="gray", vmin=0, vmax=255)
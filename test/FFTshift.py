# -*- coding: utf-8 -*-
"""
@ Time:     2024/10/12 18:49 2024
@ Author:   CQshui
$ File:     FFTshift.py
$ Software: Pycharm
"""
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import numpy as np
import cv2
import matplotlib.pyplot as plt


def fft_filter(image, x=2152, y=2164, w=212, h=124):  # x, y, w, h 分别为滤波框坐标及大小
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    mask[y:y+h, x:x+w] = 1

    fft_image = fftshift(fft2(image))

    # 对示例fft图像和U0使用掩膜
    fft_masked = fft_image * mask

    fft_masked_show = np.log(1 + np.abs(fft_masked))
    fft_masked_show = np.asarray(fft_masked_show, dtype=np.uint8)
    fft_masked_show = cv2.normalize(fft_masked_show, None, 0, 255, cv2.NORM_MINMAX, dtype=None)
    # cv2.namedWindow('image', 0)
    # cv2.imshow('image', fft_masked_show)
    # cv2.waitKey(0)

    # 滤出最中心的高亮像素块
    _, binary_image = cv2.threshold(fft_masked_show, 100, 255, cv2.THRESH_BINARY)
    # binary_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)
    binary_image_blurred = cv2.GaussianBlur(binary_image, (1, 1), 50)
    contours, _ = cv2.findContours(binary_image_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    counter_x = []
    counter_y = []
    counter_w = []
    counter_h = []
    for contour in contours:
        x0, y0, w0, h0 = cv2.boundingRect(contour)
        counter_x.append(x0)
        counter_y.append(y0)
        counter_w.append(w0)
        counter_h.append(h0)

    # 找到最大宽度的矩形并画出
    max_index = counter_w.index(max(counter_w))
    x_center = counter_x[max_index]
    y_center = counter_y[max_index]
    w_center = counter_w[max_index]
    h_center = counter_h[max_index]
    cv2.rectangle(fft_masked_show, (x_center, y_center), (x_center + w_center, y_center + h_center), (0, 255, 0), 8)

    # cv2.namedWindow('Image with Bright Spots', 0)
    # cv2.imshow("Image with Bright Spots", fft_masked_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 频谱位移到中心
    delta_x = int(0.5 * w_center + x_center - 0.5 * width)
    delta_y = int(0.5 * h_center + y_center - 0.5 * height)
    fft_masked = np.roll(fft_masked, -delta_x, axis=1)
    fft_masked = np.roll(fft_masked, -delta_y, axis=0)

    filter_image = ifft2(ifftshift(fft_masked))
    # plt.imsave('1.jpg', abs(filter_image), cmap="gray")
    # filter_image_normalized = cv2.normalize(filter_image, None, 0, 255, cv2.NORM_MINMAX, dtype=None)

    return filter_image


if __name__ == '__main__':
    image0 = cv2.imread(r'M:\Data\AutoFocusDatabase\hologram\Image__2024-04-26__20-23-57.bmp', 0)
    # image0 = cv2.imread(r'M:\Data\AutoFocusDatabase\hologram\Image__2024-04-27__20-59-30.bmp', 0)
    image1 = fft_filter(image0)

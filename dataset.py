# -*- coding: utf-8 -*-
"""
@ Time:     2024/10/12 11:30 2024
@ Author:   CQshui
$ File:     dataset.py
$ Software: Pycharm
"""
import os
import numpy as np
import torch
from utils import show_image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import albumentations as A
from reconstruct.FFTshift import fft_filter
import pandas as pd
import torch.nn.functional as F


class FigToTensor(object):
    def __call__(self, pic):
        # 将 numpy 数组 (H, W, C) 转换为 torch.Tensor (C, H, W)
        img = torch.from_numpy(pic).permute(2, 0, 1).float()
        return img


class ImageDataset(Dataset):
    def __init__(self, root_dir, csv_path, input_size, augmentation=None, transform=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.augmentation = augmentation
        self.transform = transform
        self.root_dir = root_dir
        self.hologram_path = os.path.join(self.root_dir, 'hologram')
        self.reconstruction_path = os.path.join(self.root_dir, 'reconstruction')
        self.input_size = input_size
        self.csv_path = csv_path

        # 读取CSV内容
        # row: ['532', '0.098', '0.0003600', 'Image__2024-04-26__20-23-57.bmp', 'Image__2024-04-26__20-23-57_0.0003600.jpg']
        df = pd.read_csv(csv_path)
        content_lists = [df[column].tolist() for column in df.columns]
        self.ground_zs = content_lists[2]
        self.holograms_name = content_lists[3]
        self.reconstructions_name = content_lists[4]
        # 读取当前文件夹内容
        self.files_name = os.listdir(self.hologram_path)

    def __len__(self):
        return len(self.files_name)

    def __getitem__(self, idx):
        # 加载图像及其类别标签
        hologram_name = self.files_name[idx]
        # 通过图像名称，查找位于csv中的index，索引z
        index_in_csv = self.holograms_name.index(hologram_name)
        reconstruction_name = self.reconstructions_name[index_in_csv]

        # 加载图像
        hologram = cv2.imread(os.path.join(self.hologram_path, hologram_name), 0)
        # 做一次fft滤波，滤波框大小可调，得到的结果应当是复数形式
        hologram_filtered = fft_filter(hologram)

        # 复数形式无法用CNN直接卷积，因此把实部和虚部分为2个通道
        # 分离实部和虚部
        real_part = np.real(hologram_filtered)
        imag_part = np.imag(hologram_filtered)
        # 将实部和虚部分别作为两个通道，合并为新的形状
        hologram_splited = np.stack([real_part, imag_part], axis=0)

        # 加载target和z
        reconstruction_img_pth = os.path.join(self.reconstruction_path, reconstruction_name)
        reconstruction = cv2.imread(reconstruction_img_pth, 0)
        ground_z = self.ground_zs[index_in_csv]

        # # TODO transform的形式需要重新写。
        # if self.transform:
        hologram_splited = torch.tensor(hologram_splited, dtype=torch.float32).unsqueeze(1).to(self.device)
        hologram_splited = F.interpolate(hologram_splited, size=self.input_size, mode='bilinear', align_corners=False)

        reconstruction = torch.tensor(reconstruction, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        reconstruction = F.interpolate(reconstruction, size=self.input_size, mode='bilinear', align_corners=False)

        # print(hologram_splited.shape)
        # print(reconstruction.shape)

        # 返回转换后的图像张量
        return hologram_splited, reconstruction, ground_z


def get_augmentation():
    transforms = [
          A.HorizontalFlip(p=0.5),
          A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
          A.RandomBrightnessContrast(contrast_limit=0.3, brightness_limit=0.3, p=0.2),
          A.OneOf([
                A.ImageCompression(p=0.8),
                A.RandomGamma(p=0.8),
                A.Blur(p=0.8),
            ], p=1.0),
          A.OneOf([
                A.ImageCompression(p=0.8),
                A.RandomGamma(p=0.8),
                A.Blur(p=0.8),
            ], p=1.0),
          A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_CONSTANT),
      ]

    return A.Compose(transforms)


def get_torch_transforms_3channel(img_size=224):
    data_transforms = {
        'train': transforms.Compose([
            FigToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),  # 添加 antialias=True
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation((-5, 5)),
            transforms.RandomAutocontrast(p=0.2),  # 移除 autocontrast
            transforms.Normalize([0.485]*3, [0.229]*3)  # 调整为 3 通道
        ]),
        'val': transforms.Compose([
            FigToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),  # 添加 antialias=True
            transforms.Normalize([0.485]*3, [0.229]*3)  # 调整为 3 通道
        ]),
    }
    return data_transforms


if __name__ == '__main__':
    input_size = (256, 256)
    csv = r'M:\Data\AutoFocusDatabase\AutoFocusDatabase.csv'
    train_dataset = ImageDataset(r'M:\Data\AutoFocusDatabase\train', csv, input_size, transform=None)
    x = train_dataset[0]
    hologram, reconstruction, z = x
    hologram = hologram.unsqueeze(0)
    reconstruction = reconstruction.unsqueeze(0)

    show_image(hologram)
    show_image(reconstruction)

    # train_loader = DataLoader(  # 按照批次加载训练集
    #     train_dataset, batch_size=4, shuffle=True,
    #     num_workers=0, pin_memory=True,
    # )

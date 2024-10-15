# -*- coding: utf-8 -*-
"""
@ Time:     2024/10/14 20:50 2024
@ Author:   CQshui
$ File:     pinn_test.py
$ Software: Pycharm
"""
from torch.utils.data import DataLoader
from pinn import PINNFocusNet
import torch
from utils import show_image
from dataset import ImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(train_loader, model):
    return None


if __name__ == '__main__':
    input_size = (256, 256)
    batch_size = 2

    # 数据驱动
    csv = r'M:\Data\AutoFocusDatabase\AutoFocusDatabase.csv'
    test_dataset = ImageDataset(r'M:\Data\AutoFocusDatabase\test', csv, input_size, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 加载模型
    model_read_path = r'./checkpoints/1.pth'
    model = PINNFocusNet().to(device)
    weights = torch.load(model_read_path)['state_dict']
    model.eval()
    model.to(device)

    test_loss = test(test_loader, model)

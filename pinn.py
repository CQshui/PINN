# -*- coding: utf-8 -*-
"""
@ Time:     2024/10/11 23:42 2024
@ Author:   CQshui
$ File:     pinn.py
$ Software: Pycharm
"""
import os

import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from dataset import ImageDataset
from complexcnn.modules import ComplexConv, ComplexConvTranspose, ComplexMaxPool2d, ComplexUpsample
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
from utils import accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义物理信息神经网络 (PINN)
class PINNFocusNet(nn.Module):
    def __init__(self):
        super(PINNFocusNet, self).__init__()
        self.input_size = input_size

        # 共享的特征提取层
        self.shared_layers = nn.Sequential(
            ComplexConv(1, 64, kernel_size=3, padding=1),
            ComplexMaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            ComplexConv(64, 128, kernel_size=3, padding=1),
            ComplexMaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            ComplexConv(128, 256, kernel_size=3, padding=1),
            ComplexMaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        # 计算特征提取层的输出尺寸
        self.output_size = self._calculate_output_size(input_size)
        self.flattened_size = 2 * 256 * self.output_size[0] * self.output_size[1]

        # 深度预测分支
        self.depth_branch = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)  # 预测z轴深度
        )

        # 图像重建分支
        self.image_branch = nn.Sequential(
            ComplexConvTranspose(256, 128, kernel_size=3, padding=1),
            ComplexUpsample(scale_factor=2),
            nn.ReLU(),
            ComplexConvTranspose(128, 64, kernel_size=3, padding=1),
            ComplexUpsample(scale_factor=2),
            nn.ReLU(),
            ComplexConvTranspose(64, 32, kernel_size=3, padding=1),
            ComplexUpsample(scale_factor=2),
            ComplexConvTranspose(32, 1, kernel_size=3, padding=1)
        )

    def _calculate_output_size(self, input_size):
        # 虚拟输入
        x = torch.zeros(1, 2, 1, input_size[0], input_size[1])

        # 计算特征提取层的输出尺寸
        for layer in self.shared_layers:
            x = layer(x)

        # 返回输出尺寸
        return x.size()[3:]

    def forward(self, x):
        # 共享特征提取
        shared_features = self.shared_layers(x)
        shared_flattened = shared_features.view(shared_features.size(0), -1)
        # print("shared_features shape:", shared_features.shape)  # 打印以确认形状

        # z轴深度预测
        z_pred = self.depth_branch(shared_flattened)

        # 重建聚焦图像
        image_output = self.image_branch(shared_features)  #  形状为 [4, 2, 1, 128, 128]
        # 提取实部和虚部
        real_part = image_output[:, 0, :, :, :]  #  形状为 [4, 1, 128, 128]
        imag_part = image_output[:, 1, :, :, :]  #  形状为 [4, 1, 128, 128]

        # 计算平方和
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)

        # 将结果重新组合成新的 tensor，形状为 [4, 1, 1, 128, 128]
        image_pred = magnitude.unsqueeze(1)

        return z_pred, image_pred, image_output


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

# 调整学习率
def adjust_learning_rate(optimizer, epoch, batch=0, nBatch=None):
    """ adjust learning of a given optimizer and return the new learning rate """
    new_lr = calc_learning_rate(epoch, lr, epochs, batch, nBatch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

# 计算学习率
def calc_learning_rate(epoch, init_lr, n_epochs, batch=0, nBatch=None, lr_schedule_type='cosine'):
    if lr_schedule_type == 'cosine':
        t_total = n_epochs * nBatch
        t_cur = epoch * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError('do not support: %s' % lr_schedule_type)
    return lr


# 菲涅耳衍射算子
def fresnel_propagation(I_focus, z, wavelength=532e-9, pixel_size=0.098e-6):
    """
    通过菲涅耳衍射将聚焦图像传播到z平面
    :param I_focus: 聚焦图像，形状为 [4, 2, 1, 128, 128]
    :param z: 传播距离，形状为 [4, 1]
    :param wavelength: 光波波长
    :param pixel_size: 像素大小
    :return: 在z平面传播后的全息图，形状为 [4, 2, 1, 128, 128]
    """
    N, M = I_focus.shape[-2:]
    fx = torch.fft.fftfreq(N, pixel_size).to(device)
    fy = torch.fft.fftfreq(M, pixel_size).to(device)
    FX, FY = torch.meshgrid(fx, fy)
    pi = torch.tensor(math.pi).to(device)

    # 将 z 广播到与 I_focus 相同的形状
    z = z.view(-1, 1, 1, 1, 1)  # 形状为 [4, 1, 1, 1, 1]

    # 计算传递函数 H
    H = torch.exp(1j * 2 * pi * z / wavelength * (FX ** 2 + FY ** 2) ** 0.5).to(device)

    # 提取 I_focus 的实部和虚部
    real_part = I_focus[:, 0, :, :, :].unsqueeze(1)  # 形状为 [4, 1, 1, 128, 128]
    imag_part = I_focus[:, 1, :, :, :].unsqueeze(1)  # 形状为 [4, 1, 1, 128, 128]

    # 将实部和虚部合成为复数场
    I_focus_complex = torch.complex(real_part, imag_part)

    # 计算 I_focus 的频谱
    I_focus_spectrum = torch.fft.fft2(I_focus_complex)

    # 计算传播后的频谱
    I_propagated_spectrum = I_focus_spectrum * H

    # 计算传播后的图像
    I_propagated = torch.fft.ifft2(I_propagated_spectrum)

    # 将复数结果分解为实部和虚部
    I_propagated_real = I_propagated.real
    I_propagated_imag = I_propagated.imag

    # 将实部和虚部重新组合为双通道浮点格式
    I_propagated_out = torch.cat((I_propagated_real, I_propagated_imag), dim=1)

    return I_propagated_out


# 定义损失函数：结合数据损失与物理残差
def loss_function(z_pred, z_true, reconstruction_pred, reconstruction_true, hologram, propagated_hologram, alpha=1.0, beta=1.0, gamma=1.0):
    # 深度预测损失
    depth_loss = torch.mean((z_pred - z_true*1e5) ** 2)  # 使用MSE
    # print(z_pred.item(), 1e5*z_true.item(), depth_loss.item())
    # print('depth_loss: ', depth_loss)

    # 图像重建损失：使用均方误差损失
    # print(image_pred.shape, image_true.shape)
    image_loss = F.mse_loss(reconstruction_pred, reconstruction_true)  # 使用MSE损失
    # print('image_loss: ', image_loss)

    # 物理残差损失：通过菲涅耳衍射将重建图像传播回全息图，这两张图都是 复数 形式的
    # print(hologram.shape, reconstruction_complex.shape, z_pred.shape)
    propagated_hologram = fresnel_propagation(propagated_hologram, z_true)
    # print('propagated_hologram_shape: ', propagated_hologram.shape)
    physical_residual = F.mse_loss(propagated_hologram, hologram)  # 使用MSE损失
    # print('physical_residual: ', physical_residual)

    # 总损失
    total_loss = alpha * depth_loss + beta * image_loss + gamma * physical_residual
    print(depth_loss.item(), image_loss.item(), physical_residual.item())
    # total_loss = alpha * depth_loss + beta * image_loss
    return total_loss


def train(train_loader, model, optimizer, epoch):
    metric_monitor = MetricMonitor()  # 设置指标监视器
    model.train()  # 模型设置为训练模型
    nBatch = len(train_loader)
    stream = tqdm(train_loader)
    for i, (holograms, reconstructions, zs) in enumerate(stream, start=1):  # 开始训练
        hologram = holograms.to(device, non_blocking=True)  # 加载数据
        reconstruction_true = reconstructions.to(device, non_blocking=True)  # 加载模型
        z_true = zs.unsqueeze(1).to(device, non_blocking=True)  # 形状为torch.size([4, 1])

        output = model(hologram)  # 数据送入模型进行前向传播 输入一个batch
        z_pred, reconstruction_pred, reconstruction_complex = output
        loss = loss_function(z_pred, z_true, reconstruction_pred, reconstruction_true, hologram, reconstruction_complex)

        # loss = criterion(output, target.long())  # 计算损失
        # f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
        # recall_macro = calculate_recall_macro(output, target)  # 计算recall分数
        # acc = accuracy(z_pred, z_true, reconstruction_pred, reconstruction_true, hologram, reconstruction_complex)  # 计算准确率
        metric_monitor.update('Loss', loss)  # 更新损失
        # metric_monitor.update('F1', f1_macro)  # 更新f1
        # metric_monitor.update('Recall', recall_macro)  # 更新recall
        # metric_monitor.update('Accuracy_Z', acc[0])  # 更新准确率
        # metric_monitor.update('Accuracy_R', acc[1])  # 更新准确率
        # metric_monitor.update('Accuracy_H', acc[2])  # 更新准确率
        optimizer.zero_grad()  # 清空学习率
        loss.backward()  # 损失反向传播
        optimizer.step()  # 更新优化器
        lr = adjust_learning_rate(optimizer, epoch, i, nBatch)  # 调整学习率
        stream.set_description(  # 更新进度条
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=epoch,
                metric_monitor=metric_monitor)
        )
        # 输出当前和最大内存使用情况
        # print(f"Current memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
        # print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB")
    return metric_monitor.metrics['Loss']["avg"]


def validate(val_loader, model, epoch):
    metric_monitor = MetricMonitor()  # 验证流程
    model.eval()  # 模型设置为验证格式
    stream = tqdm(val_loader)  # 设置进度条
    with torch.no_grad():  # 开始推理
        for i, (holograms, reconstructions, zs) in enumerate(stream, start=1):  # 开始训练
            hologram = holograms.to(device, non_blocking=True)  # 加载数据
            reconstruction_true = reconstructions.to(device, non_blocking=True)  # 加载模型
            z_true = zs.unsqueeze(1).to(device, non_blocking=True)

            output = model(hologram)  # 数据送入模型进行前向传播
            z_pred, reconstruction_pred, reconstruction_complex = output
            loss = loss_function(z_pred, z_true, reconstruction_pred, reconstruction_true, hologram, reconstruction_complex)

            # loss = criterion(output, target.long())  # 计算损失
            # f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
            # recall_macro = calculate_recall_macro(output, target)  # 计算recall分数
            # acc = accuracy(z_pred, z_true, reconstruction_pred, reconstruction_true, hologram, reconstruction_complex)  # 计算准确率
            metric_monitor.update('Loss', loss)  # 更新损失
            # metric_monitor.update('F1', f1_macro)  # 更新f1
            # metric_monitor.update('Recall', recall_macro)  # 更新recall
            # metric_monitor.update('Accuracy_Z', acc[0])  # 更新准确率
            # metric_monitor.update('Accuracy_R', acc[1])  # 更新准确率
            # metric_monitor.update('Accuracy_H', acc[2])  # 更新准确率
            stream.set_description(  # 更新进度条
                "Epoch: {epoch}. Validation.      {metric_monitor}".format(
                    epoch=epoch,
                    metric_monitor=metric_monitor)
            )
    return metric_monitor.metrics['Loss']["avg"]


if __name__ == "__main__":
    # 初始化网络和优化器
    batch_size = 4
    epochs = 10
    lr = 0.00001
    input_size = (128, 128)
    model_save_path = "./checkpoints"
    model = PINNFocusNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 数据驱动
    csv = r'M:\Data\AutoFocusDatabase\AutoFocusDatabase.csv'
    train_dataset = ImageDataset(r'M:\Data\AutoFocusDatabase\train', csv, input_size, transform=None)
    valid_dataset = ImageDataset(r'M:\Data\AutoFocusDatabase\val', csv, input_size, transform=None)

    train_loader = DataLoader(  # 按照批次加载训练集
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(  # 按照批次加载验证集
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # 训练循环
    for epoch in range(epochs):
        optimizer.zero_grad()
        train_loss = train(train_loader, model, optimizer, epoch)
        valid_loss = validate(val_loader, model, epoch)

        # 每隔5个epoch保存一次模型
        if epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_epoch_{epoch}.pth'))

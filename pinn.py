# -*- coding: utf-8 -*-
"""
@ Time:     2024/10/11 23:42 2024
@ Author:   CQshui
$ File:     pinn.py
$ Software: Pycharm
"""
import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from dataset import ImageDataset
from complexcnn.modules import ComplexConv
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义物理信息神经网络 (PINN)
class PINNFocusNet(nn.Module):
    def __init__(self):
        super(PINNFocusNet, self).__init__()
        # 共享的特征提取层
        self.shared_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # 深度预测分支
        self.depth_branch = nn.Sequential(
            nn.Linear(256 * 64 * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)  # 预测z轴深度
        )
        # 图像重建分支
        self.image_branch = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # 共享特征提取
        shared_features = self.shared_layers(x)
        shared_flattened = shared_features.view(shared_features.size(0), -1)
        # print("shared_features shape:", shared_features.shape)  # 打印以确认形状

        # z轴深度预测
        z_pred = self.depth_branch(shared_flattened)

        # 重建聚焦图像
        image_pred = self.image_branch(shared_features)

        return z_pred, image_pred


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
    :param I_focus: 聚焦图像
    :param z: 传播距离
    :param wavelength: 光波波长
    :param pixel_size: 像素大小
    :return: 在z平面传播后的全息图
    """
    N, M = I_focus.shape[-2:]
    fx = torch.fft.fftfreq(N, pixel_size).to(device)
    fy = torch.fft.fftfreq(M, pixel_size).to(device)
    FX, FY = torch.meshgrid(fx, fy)
    pi = torch.tensor(math.pi).to(device)
    H = torch.exp(1j * 2 * pi * z / wavelength * (FX ** 2 + FY ** 2) ** 0.5).to(device)
    I_focus_spectrum = torch.fft.fft2(I_focus).to(device)
    I_propagated_spectrum = I_focus_spectrum * H
    I_propagated = torch.fft.ifft2(I_propagated_spectrum).to(device)

    return torch.abs(I_propagated)


# 定义损失函数：结合数据损失与物理残差
def loss_function(z_pred, z_true, image_pred, image_true, hologram, alpha=1.0, beta=1.0, gamma=1.0):
    # 深度预测损失
    depth_loss = torch.mean((z_pred - z_true) ** 2)  # 仍然使用MSE

    # 图像重建损失：使用交叉熵损失
    image_loss = F.cross_entropy(image_pred, image_true)  # 交叉熵损失

    # 物理残差损失：通过菲涅耳衍射将重建图像传播回全息图
    propagated_hologram = fresnel_propagation(image_pred, z_pred)
    # physical_residual = torch.mean((propagated_hologram - hologram) ** 2)
    physical_residual = F.cross_entropy(propagated_hologram, hologram)

    # 总损失
    total_loss = alpha * depth_loss + beta * image_loss + gamma * physical_residual
    return total_loss


def train(train_loader, model, optimizer, epoch):
    metric_monitor = MetricMonitor()  # 设置指标监视器
    model.train()  # 模型设置为训练模型
    nBatch = len(train_loader)
    stream = tqdm(train_loader)
    for i, (holograms, reconstructions, zs) in enumerate(stream, start=1):  # 开始训练
        hologram = holograms.to(device, non_blocking=True)  # 加载数据
        reconstruction = reconstructions.to(device, non_blocking=True)  # 加载模型
        z = zs.to(device, non_blocking=True)

        output = model(hologram)  # 数据送入模型进行前向传播 todo 输入的量都是单张图像还是一个batch？
        z_pred, reconstruction_pred = output
        loss = loss_function(z_pred, z, reconstruction_pred, reconstruction, hologram)

        # loss = criterion(output, target.long())  # 计算损失
        # f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
        # recall_macro = calculate_recall_macro(output, target)  # 计算recall分数
        # acc = accuracy(output, target)  # 计算准确率分数
        metric_monitor.update('Loss', loss)  # 更新损失
        # metric_monitor.update('F1', f1_macro)  # 更新f1
        # metric_monitor.update('Recall', recall_macro)  # 更新recall
        # metric_monitor.update('Accuracy', acc)  # 更新准确率
        optimizer.zero_grad()  # 清空学习率
        loss.backward()  # 损失反向传播
        optimizer.step()  # 更新优化器
        lr = adjust_learning_rate(optimizer, epoch, i, nBatch)  # 调整学习率
        stream.set_description(  # 更新进度条
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=epoch,
                metric_monitor=metric_monitor)
        )
    return metric_monitor.metrics['Loss']["avg"]  # 返回结果


def validate(val_loader, model, epoch):
    metric_monitor = MetricMonitor()  # 验证流程
    model.eval()  # 模型设置为验证格式
    stream = tqdm(val_loader)  # 设置进度条
    with torch.no_grad():  # 开始推理
        for i, (holograms, reconstructions, zs) in enumerate(stream, start=1):  # 开始训练
            hologram = holograms.to(device, non_blocking=True)  # 加载数据
            reconstruction = reconstructions.to(device, non_blocking=True)  # 加载模型
            z = zs.to(device, non_blocking=True)

            output = model(hologram)  # 数据送入模型进行前向传播 todo 输入的量都是单张图像还是一个batch？
            z_pred, reconstruction_pred = output
            loss = loss_function(z_pred, z, reconstruction_pred, reconstruction, hologram)

            # loss = criterion(output, target.long())  # 计算损失
            # f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
            # recall_macro = calculate_recall_macro(output, target)  # 计算recall分数
            # acc = accuracy(output, target)  # 计算准确率分数
            metric_monitor.update('Loss', loss)  # 更新损失
            # metric_monitor.update('F1', f1_macro)  # 更新f1
            # metric_monitor.update('Recall', recall_macro)  # 更新recall
            # metric_monitor.update('Accuracy', acc)  # 更新准确率
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
    lr = 0.0001
    model = PINNFocusNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 数据驱动
    csv = r'M:\Data\AutoFocusDatabase\AutoFocusDatabase.csv'
    train_dataset = ImageDataset(r'M:\Data\AutoFocusDatabase\train', csv, transform=None)
    valid_dataset = ImageDataset(r'M:\Data\AutoFocusDatabase\val', csv, transform=None)

    train_loader = DataLoader(  # 按照批次加载训练集
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(  # 按照批次加载验证集
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )


    # hologram = torch.randn(1, 1, 64, 64).to(device)  # 输入全息图
    # z_true = torch.tensor([0.01]).to(device)  # 真实z轴深度
    # image_true = torch.randn(1, 1, 64, 64).to(device)  # 真实聚焦图像

    # 训练循环
    for epoch in range(epochs):
        optimizer.zero_grad()
        train_loss = train(train_loader, model, optimizer, epoch)
        valid_loss = validate(val_loader, model, epoch)

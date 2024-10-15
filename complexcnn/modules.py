# -*- coding: utf-8 -*-
"""
@ Time:     2024/10/13 0:16 2024
@ Author:   CQshui
$ File:     modules.py
$ Software: Pycharm
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ComplexConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):  # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.conv_re(x[:, 0]) - self.conv_im(x[:, 1])
        imaginary = self.conv_re(x[:, 1]) + self.conv_im(x[:, 0])
        output = torch.stack((real, imaginary), dim=1)
        return output


class ComplexConvTranspose(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConvTranspose, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding

        ## Model components
        self.conv_transpose_re = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                                    output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_transpose_im = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                                    output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):  # shape of x: [batch, 2, channel, axis1, axis2]
        real = self.conv_transpose_re(x[:, 0]) - self.conv_transpose_im(x[:, 1])
        imaginary = self.conv_transpose_re(x[:, 1]) + self.conv_transpose_im(x[:, 0])
        output = torch.stack((real, imaginary), dim=1)
        return output


class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(ComplexMaxPool2d, self).__init__()
        self.maxpool_re = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
        self.maxpool_im = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, x):  # shape of x: [batch, 2, channel, axis1, axis2]
        real = self.maxpool_re(x[:, 0])
        imaginary = self.maxpool_im(x[:, 1])
        output = torch.stack((real, imaginary), dim=1)
        return output


class ComplexAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super(ComplexAvgPool2d, self).__init__()
        self.avgpool_re = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)
        self.avgpool_im = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)

    def forward(self, x):  # shape of x: [batch, 2, channel, axis1, axis2]
        real = self.avgpool_re(x[:, 0])
        imaginary = self.avgpool_im(x[:, 1])
        output = torch.stack((real, imaginary), dim=1)
        return output


class ComplexUpsample(nn.Module):
    def __init__(self, scale_factor, mode='bilinear'):
        super(ComplexUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):  # shape of x: [batch, 2, channel, axis1, axis2]
        real = F.interpolate(x[:, 0], scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        imaginary = F.interpolate(x[:, 1], scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        output = torch.stack((real, imaginary), dim=1)
        return output


if __name__ == "__main__":
    ## Random Tensor for Input
    ## shape : [batchsize,2,channel,axis1_size,axis2_size]
    ## Below dimensions are totally random
    x = torch.randn((10, 2, 3, 100, 100))

    # 1. Make ComplexConv Object
    ## (in_channel, out_channel, kernel_size) parameter is required
    complexConv = ComplexConv(3, 10, (5, 5))

    # 2. compute
    y = complexConv(x)

    # 3. 反卷积
    complexTranspose = ComplexConvTranspose(10, 3, (5, 5))
    z = complexTranspose(y)
    print(x)
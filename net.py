import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import xZero
# --------------------搭建网络--------------------------------


class SeparableConv2d(nn.Module):  # Depth wise separable conv
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        # 每个input channel被自己的filters卷积操作
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            SeparableConv2d(270, 32, 3, 1, 1),  # 3*3卷积核
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.conv2 = nn.Sequential(
            SeparableConv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3)
        )
        self.conv3 = nn.Sequential(
            SeparableConv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3)
        )
        self.conv4 = nn.Sequential(
            SeparableConv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3)
        )
        self.conv5 = nn.Sequential(
            SeparableConv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 9 * 9, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 16),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)  # 展平
        output = self.classifier(x)
        return output


    def numFeatures(self, x):
        size = x.size()[1:]  # 获取卷积图像的h,w,depth
        num = 1
        for s in size:
            num *= s
            # print(s)
        return num

    def init_weights(self):  # 初始化权值
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
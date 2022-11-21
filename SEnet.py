# https://blog.csdn.net/weixin_44791964/article/details/121371986
"""
空间注意力机制
通道注意力机制： 比如 0.9点、线>0.1明暗特征
"""

# SEnet通道注意力机制

import torch
from torch import nn


class SEnet(nn.Module):
    def __init__(self, input_channels, ratio=16):
        super(SEnet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_channels, input_channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(input_channels // ratio, input_channels, bias=False),
            nn.Sigmoid(),  # 获得每个通道的权重
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # b, c, h, w -> b, c, 1, 1 -> b, c
        avg = self.avg_pool(x).view([b, c])
        # b, c, 1, 1
        fc = self.fc(avg).view([b, c, 1, 1])
        return x * fc


inputs = torch.ones([2, 512, 32, 32])
net = SEnet(512)
outputs = net(inputs)
print(outputs)

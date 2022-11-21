# ECAnet: 通道注意力机制
import torch
from torch import nn
import math


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gammma=2):
        super(eca_block, self).__init__()
        kernal_size = int(abs((math.log(channel, 2) + b) / gammma))
        kernel_size = kernal_size if kernal_size % 2 else kernal_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        padding = kernal_size // 2
        self.conv = nn.Conv1d(1, 1, kernel_size,padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = self.avg_pool(x).view([b, 1, c])
        out = self.conv(avg_pool)
        out = self.sigmoid(out).view([b, c, 1, 1])

        return out * x


net = eca_block(512)
# print(net)
inputs = torch.ones([2, 512, 26, 26])
outputs = net(inputs)
print(outputs)

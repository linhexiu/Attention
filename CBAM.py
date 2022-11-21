# CBAM将通道注意力机制和空间注意力机制进行一个结合

# channel attention module + spatial attention module
# channel attention modul 中对w,h维度进行最大化和平均池化，也就是说活了每个通道（特征）里的最大的值和平均值
# 该特征条是包含所有全局特征的
# spatial attention module 中对c维度进行最大和平均池化，也就是说只保留最大和平均的通道（特征）
# 该特征尺寸是w*h*1（压扁通道数）

import torch
from torch import nn


class channel_attention(nn.Module):
    def __init__(self, input_channels, ratio=16):
        super(channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(input_channels, input_channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(input_channels // ratio, input_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out = self.max_pool(x).view([b, c])
        avg_poop_out = self.avg_pool(x).view([b, c])

        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_poop_out)

        out = max_fc_out + avg_fc_out

        out = self.sigmoid(out).view([b, c, 1, 1])

        return out * x


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)

        pool_out = torch.cat([max_pool_out, avg_pool_out], dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)

        return out * x


class Cbam(nn.Module):
    def __init__(self, input_channels, ratio=16, kernel_size=7):
        super(Cbam, self).__init__()
        self.channel_attention = channel_attention(input_channels, ratio)
        self.spatial_attention = spatial_attention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


net = Cbam(512)
# print(net)
inputs = torch.ones([2, 512, 26, 26])
outputs = net(inputs)
print(outputs)

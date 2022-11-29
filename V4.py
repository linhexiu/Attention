# https://www.cnblogs.com/alexme/p/11366792.html
#  简单卷积网络模型的可视化
#  手动设置4个过滤器（feature maps）
import cv2
import matplotlib.pyplot as plt

img_path = 'img.png'
bgr_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# b, g, r = cv2.split(bgr_img)
# plt.figure(figsize=[20, 5])
#
# plt.subplot(141)
# plt.imshow(r, cmap='gray')
# plt.title('red channel')
# plt.subplot(142)
# plt.imshow(g, cmap='gray')
# plt.title('green channel')
# plt.subplot(143)
# plt.imshow(b, cmap='gray')
# plt.title('blue channel')
#
# imgMerge = cv2.merge((r, g, b))
# plt.subplot(144)
# plt.imshow(imgMerge)
# plt.title('merge output')

gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
gray_img = gray_img.astype('float32') / 255
# plt.imshow(gray_img, cmap='gray')
# plt.title('gray output')
#
# plt.show()

import numpy as np

filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])

filter_1 = filter_vals
filter_2 = -filter_vals
filter_3 = filter_1.T
filter_4 = -filter_3

filters = np.array([filter_1, filter_2, filter_3, filter_4])
fig = plt.figure(figsize=(10, 5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i + 1))
    width, height = filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(filters[i][x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if filters[i][x][y] < 0 else 'black')
# 显示核的图像
plt.show()

import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self, weight):
        super(Net, self).__init__()
        k_height, k_width = weight.shape[2:]
        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)
        self.pool = nn.MaxPool2d(4, 4)

    def forward(self, x):
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)
        pooled_x = self.pool(activated_x)

        return conv_x, activated_x, pooled_x


weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)
print('Filters shape: ', filters.shape)
print('weights shape: ', weight.shape)
print(model)


def viz_layer(layer, n_filters=4):
    fig = plt.figure(figsize=(20, 20))
    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(layer[0, i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i + 1))
    plt.show()


plt.imshow(gray_img, cmap='gray')
# 原来图片的灰度图
plt.show()

# 为 gray img 添加 1 个 batch 维度，以及 1 个 channel 维度，并转化为 tensor
gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)
print(gray_img.shape)
print(gray_img_tensor.shape)

conv_layer, activated_layer, pooled_layer = model(gray_img_tensor)
viz_layer(conv_layer)
viz_layer(activated_layer)
viz_layer(pooled_layer)

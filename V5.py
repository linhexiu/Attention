# https://www.cnblogs.com/alexme/p/11366792.html
# 多层卷积网络模型的可视化

import torch
import torchvision

from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

data_transform = transforms.ToTensor()

train_data = FashionMNIST(root='./data', train=True, download=True, transform=data_transform)
test_data = FashionMNIST(root='./data', train=False, download=True, transform=data_transform)
print('Train data, number of images: ', len(train_data))
print('Test data, number of images: ', len(test_data))

batch_size = 20
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 可视化部分数据集

# import numpy as np
# import matplotlib.pyplot as plt
#
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
# images = images.numpy()
#
# fig = plt.figure(figsize=(25, 4))
# for idx in np.arange(batch_size):
#     ax = fig.add_subplot(2, int(batch_size / 2), idx + 1, xticks=[], yticks=[])
#     ax.imshow(np.squeeze(images[idx]), cmap='gray')
#     ax.set_title(classes[labels[idx]])
# plt.show()

# 定义模型
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入通道为1，输出通道为16
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(32 * 7 * 7, 24)
        self.out = nn.Linear(24, 10)
        self.dropout = nn.Dropout(0.5)  # dropout 在一定程度上起到防止过拟合的作用
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        x = self.dropout(x)
        x = self.softmax(self.out(x))

        return x


# 训练模型

import torch.optim as optim

net = Net()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters())


def train(n_epochs):
    for epoch in range(n_epochs):
        running_loss = 0.0
        for batch_i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_i % 1000 == 999:
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 1000))
                running_loss = 0.0

    print('Finished Training')


#
# n_epochs = 10
#
# train(n_epochs)
#
# model_dir = 'saved_models/'
# model_name = 'model_best.pt'
#
# torch.save(net.state_dict(), model_dir + model_name)

# 加载训练的模型
# net.load_state_dict(torch.load('saved_models/model_best.pt'))
#
# print(net)
#
# # 在测试数据集上测试模型
# test_loss = torch.zeros(1)
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
#
# print(class_correct)
# print(test_loss)
#
# net.eval()
# for batch_i, data in enumerate(test_loader):
#     inputs, labels = data
#     outputs = net(inputs)
#     loss = criterion(outputs, labels)
#
#     test_loss = test_loss + ((torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))
#     _, predicted = torch.max(outputs.data, 1)
#     correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))
#
#     for i in range(batch_size):
#         label = labels.data[i]
#         class_correct[label] += correct[i].item()
#         class_total[label] += 1
#
# print('Test Loss: {:.6f}\n'.format(test_loss.numpy()[0]))
# for i in range(10):
#     if class_total[i] > 0:
#         print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
#             classes[i], 100 * class_correct[i] / class_total[i],
#             np.sum(class_correct[i]), np.sum(class_total[i])))
#     else:
#         print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
#
# print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
#     100. * np.sum(class_correct) / np.sum(class_total),
#     np.sum(class_correct), np.sum(class_total)))

# 特征可视化
dataiter = iter(test_loader)
images, labels = dataiter.next()
images = images.numpy()

idx = 15
img = np.squeeze(images[idx])

import cv2
import matplotlib.pyplot as plt

# 原图
plt.imshow(img, cmap='gray')
plt.show()

# conv1可视化
weights = net.conv1.weight.data
w = weights.numpy()
print(w.shape)  # (16, 1, 3, 3) 16个 feature map

fig = plt.figure(figsize=(30, 10))
columns = 4 * 2
row = 4
for i in range(0, columns * row):
    fig.add_subplot(row, columns, i + 1)
    if ((i % 2) == 0):
        plt.imshow(w[int(i / 2)][0], cmap='gray')
    else:
        c = cv2.filter2D(img, -1, w[int((i - 1) / 2)][0])
        plt.imshow(c, cmap='gray')
plt.show()

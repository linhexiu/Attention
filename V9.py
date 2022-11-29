# 可视化神经网络的过滤器
# 想要观察卷积神经网络学到的过滤器
# 显示每个过滤器所响应的视觉模式

from keras.applications import VGG16
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

# import tensorflow as tf
#
# tf.compat.v1.disable_eager_execution()

model = VGG16(weights='imagenet',
              include_top=False)

model.summary()

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

# 获取损失相对于输入的梯度
grads = K.gradients(loss, model.input)[0]
# 梯度标准化技巧
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# 给定numpy输入值，得到numpy输出值
iterate = K.function([model.input], [loss, grads])

# 通过随机梯度下降让损失最大化
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
step = 1.
for i in range(40):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step


# 将张量转换为有效图像的实用函数
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    # x *= 255
    # x = np.clip(x, 0, 255)
    # x/=255.
    return x


# 生成过滤器可视化的函数
# 构建一个损失函数，将该层第 n 个过滤器的激活最大化
def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)


# block3_conv1 层第 0 个过滤器响应的是波尔卡点（polka-dot）图案
plt.imshow(generate_pattern('block3_conv1', 0))
plt.show()
# 　生成某一层中所有过滤器响应模式组成的网格
# 查看如下5个层的过滤器模式
layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
for layer_name in layer_names:
    # 显示通道中的前64个滤波器
    size = 64
    margin = 5
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
    for i in range(8):
        for j in range(8):
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
    plt.figure(figsize=(20, 20))
    plt.imshow(results)
    plt.show()

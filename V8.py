from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
from keras.preprocessing import image as kimage
import numpy as np

if __name__ == '__main__':
    # 加载保存的模型
    model = models.load_model('cats_and_dogs_small_1.h5')
    model.summary()

    # 加载一张猫的图像
    img = kimage.load_img(path='./dataset/training_set/cats/cat.1700.jpg', target_size=(150, 150))
    img_tensor = kimage.img_to_array(img)
    # print(img_tensor.shape) # (150, 150, 3)
    img_tensor = img_tensor.reshape((1,) + img_tensor.shape)
    # print(img_tensor.shape) # (1, 150, 150, 3)
    img_tensor /= 255.
    # print(img_tensor[0].shape) # (150, 150, 3)
    plt.imshow(img_tensor[0])
    plt.show()

    # 提取前8层的输出
    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    # 以预测模式运行模型 activations包含卷积层的8个输出
    activations = activation_model.predict(img_tensor)
    print(len(activations))  # 8
    print(activations[0].shape)  # (1, 148, 148, 32)

    first_layer_activation = activations[0]
    plt.matshow(first_layer_activation[0, :, :, 9], cmap='viridis')
    plt.show()
    # 清空当前图像
    # plt.clf()

    # 将每个中间激活的所有通道可视化
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)
        img_per_row = 16
        for layer_name, layer_activation in zip(layer_names, activations):
            n_features = layer_activation.shape[-1]  # 32
            size = layer_activation.shape[1]  # 148
            n_cols = n_features // img_per_row
            display_grid = np.zeros((size * n_cols, img_per_row * size))
            for col in range(n_cols): # 行
                for row in range(img_per_row): # 列
                    channel_image = layer_activation[0, :, :, col * img_per_row + row]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

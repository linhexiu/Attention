from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
from keras.preprocessing import image as kimage

'''
rotation_range 是角度值（在 0~180 范围内），表示图像随机旋转的角度范围。
width_shift 和 height_shift 是图像在水平或垂直方向上平移的范围（相对于总宽
度或总高度的比例）。
shear_range 是随机错切变换的角度。
zoom_range 是图像随机缩放的范围。
horizontal_flip 是随机将一半图像水平翻转。如果没有水平不对称的假设（比如真
实世界的图像），这种做法是有意义的。
fill_mode是用于填充新创建像素的方法，这些新像素可能来自于旋转或宽度/高度平移

'''
# 训练样本的目录
train_dir = './dataset/training_set/'
# 验证样本的目录
validation_dir = './dataset/validation_set/'
# 测试样本目录
test_dir = './dataset/test_set/'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(150, 150),
    class_mode='binary',
    batch_size=20
)

# 验证样本生成器
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = train_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(150, 150),
    class_mode='binary',
    batch_size=20
)

# 测试样本生成器
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = train_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(150, 150),
    class_mode='binary',
    batch_size=20
)
if __name__ == '__main__':
    data, lables = next(train_generator)
    print(data.shape)  # (20, 128, 128, 3)
    print(lables.shape)  # (20,)
    # 查看其中一张图像以及其标签
    img_test = Image.fromarray((255 * data[0]).astype('uint8'))
    # img_test.show()
    # print(lables[0])

    # 构建训练网络
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    # 添加一个dropout层可进一步提高识别率
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(units=512, activation='relu'))
    model.add(layers.Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    model.summary()

    # 拟合模型
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50
    )
    model.save('cats_and_dogs_small_1.h5')
    # 测试测试集的准确率
    test_eval = model.evaluate_generator(test_generator)
    print(test_eval)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

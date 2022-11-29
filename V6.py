# https://keras-lx.blog.csdn.net/article/details/122926602?spm=1001.2014.3001.5502
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt

# 训练样本的目录
train_dir = './dataset/training_set/'
# 验证样本的目录
validation_dir = './dataset/validation_set/'
# 测试样本目录
test_dir = './dataset/test_set/'
# 训练样本生成器
train_datagen = ImageDataGenerator(rescale=1. / 255)
# 由于是二分类 class_mode='binary'
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(128, 128),
    class_mode='binary',
    batch_size=20
)

# 验证样本生成器
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = train_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(128, 128),
    class_mode='binary',
    batch_size=20
)

# 测试样本生成器
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = train_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(128, 128),
    class_mode='binary',
    batch_size=20
)

if __name__ == '__main__':
    data, lables = next(train_generator)
    print(data.shape)  # (20, 128, 128, 3)
    print(lables.shape)  # (20,)
    # 查看其中一张图像以及其标签
    # img_test = imagenet1k_classes.fromarray((255 * data[0]).astype('uint8'))
    # img_test.show()
    # print(lables[0])

    # 构建训练网络
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
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
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50
    )

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

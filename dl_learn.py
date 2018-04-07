from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras.applications.vgg16 import VGG16


num_classes = 10
img_height, img_width = 64, 64

# VGG19モデルの作成
def VGG19():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

# VGG19モデルにBatch Normalizationを入れてみる...
def batchVgg():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

# VGG16でFine Tuneingを試してみる
def fineTune():
    input_tensor = Input(shape=(img_width, img_height, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # FC層を構築
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(2560, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_classes, activation='softmax'))

    # VGG16とFCを接続
    model = Model(input=vgg16.input, output=top_model(vgg16.output))

    # 最後のconv層の直前までの層をfreeze
    for layer in model.layers[:15]:
        layer.trainable = False
    
    return model

# VGG16でFine Tuneingを試してみる, その2
# （結局利用せず,,,)
def fineTune2():
    vgg = VGG16(weights='imagenet', include_top=True)
    w = vgg.layers[-2].get_weights()

    model = Sequential(vgg.layers[:-1])
    for l in model.layers:
        l.trainable = False
    model.add(Dense(num_classes, activation='softmax'))

    return model
    # model.compile(optimizer='sgd', loss='binary_crossentropy')
    # model.fit(x_train, y_train, batch_size=4, epochs=1)

    # model.layers[-2].trainable = True
    # model.compile(optimizer='sgd', loss='binary_crossentropy')
    # model.fit(x_train, y_train, batch_size=4, epochs=1)


def main():
    # モデルの作成
    model = fineTune()

    # 交差クロスエントロピー、SGDを利用
    model.compile(loss='categorical_crossentropy',
                  # optimizer=Adam(lr=0.001),
                  optimizer=SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
  　
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        featurewise_center=True,
        featurewise_std_normalization=True,
        zca_whitening=True
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
    )
  
    train_generator = train_datagen.flow_from_directory(
        '/my_data/train',
        target_size=(img_height, img_width),
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        '/my_data/test',
        target_size=(img_height, img_width),
        class_mode='categorical')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=70,
        epochs=500,
        validation_data=test_generator,
        validation_steps=70,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=0)]
    )

    model.save('/output/vg19.h5')

    

if __name__ == '__main__':
    main()

"""
Title: Simple MNIST convnet
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2015/06/19
Last modified: 2020/04/21
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
"""

"""
## Setup
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.keras.models import load_model
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array

"""
## 機能の切り替えフラグ
##
## モデルの学習を行う: learnModel => True 
## 学習したモデルを保存する: saveModel => True  *モデルの保存は learnModel と saveModel が True の時に行う
## 学習済みモデルを使用する: useLearnedModel => True 
## モデルの評価を行う: evaluateTrainedModel => True 
## 取り込んだ画像を表示する: showFigureTest => True  とりあえず3x3で表示するように組んでいる
## データ拡張を行う: generateFigure => True  
## データ拡張した画像を保存する: saveNewFigure => True  *データ拡張することが前提なので、generateFigureをtrueにすること。Falseで加工した画像を9枚表示する
##
"""
# #################################################################################
# learnModel = True
# saveModel = True
# useLearnedModel = False
# evaluateTrainedModel = False
# showFigureTest = False
# generateFigure = False
# saveNewFigure = False
# useLearnedModelForLearing = False
# ###################################################################################

###################################################################################
learnModel = False
saveModel = False
useLearnedModel = True
evaluateTrainedModel =  True
showFigureTest = False
generateFigure = False
saveNewFigure = False
useLearnedModelForLearing = False
###################################################################################


"""
## データの読み込み
"""
# Model / data parameters
num_classes = 2
input_shape = (256, 256, 3)

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

train_ds = image_dataset_from_directory(
    directory='training_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))
validation_ds = image_dataset_from_directory(
    directory='validation_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,

    image_size=(256, 256))

# image_dataset_from_directoryのshuffleをTrueにするとデータをシャッフルするが、test_dsを使用する度にデータをシャッフルするため、
# 正解データのラベルとデータを分けたあとにtest_dsを呼び出すと、データとラベルの関係が崩れるので要注意
test_ds = image_dataset_from_directory(
    directory='test_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256),
    shuffle=False,
    validation_split=None
    )


"""
## Build the model
"""
if learnModel:
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    """
    ## Train the model
    """
    batch_size = 128
    epochs = 5
    if useLearnedModelForLearing:
        model = load_model('learned_model.h5')
    else:
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy","Precision","AUC"])
    model.fit(train_ds, batch_size=batch_size, epochs=epochs, validation_data=validation_ds)


"""
## 学習済みモデルを読み込む
## 学習済みモデルはfile.pyと同じファイルに保存すること
"""
if useLearnedModel:
    model = load_model('learned_model.h5')


"""
## Evaluate the trained model
"""
if evaluateTrainedModel:
    # testデータを準備したあとで使用
    score = model.evaluate(test_ds, verbose=0)
    print("@@@@@@@@@@@@@@@@@ Result Test @@@@@@@@@@@@@@@")
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print("Test Precision:", score[2])
    print("Test AUC:", score[3])
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


    """
    ## コンフュージョンマトリクスのための処理
    """

    test_np = tfds.as_numpy(test_ds)
    y_preds = np.empty((2,2), int)
    y_pred_argmax_datas = np.empty(0, int)
    y_test_argmax_datas = np.empty(0, int)
    for i in range(len(test_np)):
        print(i)

        # y_test: テストのラベル
        # x_test: テストの画像データ
        # test_npの形 [(128, 128, 3, 32), (128, 128, 3, 32), (128, 128, 3, 32)]
        # https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
        y_test = [x[1] for x in test_np][i]
        x_test = [x[0] for x in test_np][i]


        y_pred = model.predict(x_test)
        y_pred_argmax = tf.argmax(y_pred, axis = 1).numpy()
        y_test_argmax = tf.argmax(y_test, axis = 1).numpy()

        if i == 0:
            y_preds = y_pred
            y_pred_argmax_datas = y_pred_argmax
            y_test_argmax_datas = y_test_argmax
        else:
            y_preds = np.append(y_preds, y_pred, axis=0)
            y_pred_argmax_datas = np.append(y_pred_argmax_datas,y_pred_argmax)
            y_test_argmax_datas = np.append(y_test_argmax_datas,y_test_argmax)

    print("@@@@@@@@@@@@@@@@@ y_pred_argmax_datas @@@@@@@@@@@@@@@")
    print(y_pred_argmax_datas)
    print("")
    print("@@@@@@@@@@@@@@@@@ y_test_argmax_datas @@@@@@@@@@@@@@@")
    print(y_test_argmax_datas)
    print("")
    print("@@@@@@@@@@@@@@@@@ y_pred_argmax_data @@@@@@@@@@@@@@@")

    class1 = 0
    class2 = 0
    for i in range(len(y_pred_argmax_datas)):
        if y_test_argmax_datas[i] == 0:
            class1 += 1
            print("class1",class1, "  ", y_pred_argmax_datas[i], "  ", y_preds[i])
        elif y_test_argmax_datas[i] == 1:
            class2 += 1
            print("class2",class2, "  ", y_pred_argmax_datas[i], "  ", y_preds[i])

    print("")
    print("@@@@@@@@@@@@@@@@@ Result matrix @@@@@@@@@@@@@@@")
    print(tf.math.confusion_matrix(y_test_argmax_datas, y_pred_argmax_datas))



"""
## 学習したモデルを保存する
"""
if learnModel & saveModel:
    model.save('learned_model.h5')
    print("")
    print("finish saving the model!")
    print("")




"""
## 取り込んだ画像を表示させる
"""
if showFigureTest:

    class_names = test_ds.class_names
    plt.figure(figsize=(10, 10))
    for images, labels in test_ds:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            #軸を表示しない
            plt.xticks(color = "None")
            plt.yticks(color = "None")
            plt.tick_params(bottom = False, left = False)
            #表示
            plt.imshow(images[i].numpy().astype("uint8"))            
            plt.title(class_names[tf.math.argmax(labels[i]).numpy()])
            plt.axis("off")
    plt.show()



"""
## ImageDataGeneratorを使用したデータ拡張
"""
if generateFigure:

    """
    ## ImageDataGeneratorの設定
    ## 使用するdatageneratorのコメントアウトを外して使用すること
    """
    # # 回転させる
    # datagen = ImageDataGenerator(rotation_range = 60)
    # # ランダムに上下反転する。
    # datagen = image.ImageDataGenerator(vertical_flip=True) 
    # # ランダムに左右反転する。
    # datagen = image.ImageDataGenerator(horizontal_flip=True)
    # [-0.3 * Height, 0.3 * Height] の範囲でランダムに上下平行移動する。
    # datagen = image.ImageDataGenerator(height_shift_range=0.7)
    # # [-0.3 * Width, 0.3 * Width] の範囲でランダムに左右平行移動する。
    # datagen = image.ImageDataGenerator(width_shift_range=0.6)
    # # -5° ~ 5° の範囲でランダムにせん断する。 
    # datagen = image.ImageDataGenerator(shear_range=5)
    # # [1 - 0.3, 1 + 0.3] の範囲でランダムに拡大縮小する。
    # datagen = image.ImageDataGenerator(zoom_range=0.3)
    # # [-5.0, 5.0] の範囲でランダムに画素値に値を足す。
    # datagen = image.ImageDataGenerator(channel_shift_range=5.)
    # # [0.3, 1.0] の範囲でランダムに明度を変更する。
    # datagen = ImageDataGenerator(brightness_range=[0.3, 0.4])
    # datagen = ImageDataGenerator(channel_shift_range = 100)

    datagen = ImageDataGenerator(rotation_range = 40, vertical_flip=True,horizontal_flip=True,height_shift_range=0.3, width_shift_range=0.3,channel_shift_range = 100)


    """
    ## 作成した画像の保存先の設定
    """
    # 指定したディレクトリが存在しないとエラーになるので、
    # 予め作成しておく。
    save_path = 'output'
    import os
    os.makedirs(save_path, exist_ok=True)


    """
    ## 画像の生成と生成した画像の保存を行う
    """
    class_names = test_ds.class_names
    plt.figure(figsize=(10, 10))
    for images, labels in test_ds:

        rangeNumber = len(images) if saveNewFigure else 9
        for i in range(rangeNumber):

            # (Height, Width, Channels)  -> (1, Height, Width, Channels) 
            img=images[i][np.newaxis, :, :, :]

            # ImageDataGeneratorを適用、1枚しかないので、ミニバッチ数は1
            if saveNewFigure:
                gen = datagen.flow(img.numpy(), batch_size=1, save_to_dir=save_path, save_prefix='generated', save_format='png')
                batches = next(gen)
            else:
                gen = datagen.flow(img.numpy(), batch_size=1)
                # next関数を使う、なぜ使うかはわかっていない
                batches = next(gen) 
                # 画像として表示するため、3次元データにし、float から uint8 にキャストする。
                gen_img = batches[0].astype(np.uint8)

                ax = plt.subplot(3, 3, i + 1)
                # 軸を表示しない
                plt.xticks(color = "None")
                plt.yticks(color = "None")
                plt.tick_params(bottom = False, left = False)
                # 表示
                plt.imshow(gen_img)
                
                plt.title(class_names[tf.math.argmax(labels[i]).numpy()])
                plt.axis("off")

    if not saveNewFigure:
        plt.show()
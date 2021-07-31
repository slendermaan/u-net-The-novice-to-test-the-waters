# coding=utf-8
import matplotlib
from keras.optimizers import Adam
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers.merge import concatenate
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
import os
import  keras
from keras import backend as K
from tqdm import tqdm
seed = 7
os.environ['KMP_DUPLICATE_LIB_OK']='True'
np.random.seed(seed)

img_w = 256
img_h = 256
n_label = 1
classes = [0., 1.]


smooth = 1.  # 用于防止分母为0.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)  # 将 y_true 拉伸为一维.
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def load_img(path, grayscale):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float") / 255.0
    return img


filepath = 'F:\新建文件夹 - 副本 (2)\赛道B附件\data\\train'


def get_train_val(val_rate=0.25):
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(filepath + '\src'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set


# data for training
def generateData(batch_size, data=[]):
    # print 'generateData...'
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            #img = load_img("F:\新建文件夹 - 副本 (2)\赛道B附件\data\\train\src\\748.png",grayscale=False)
            img = Image.open(filepath + '\src\\' + url).convert('RGB')
            x=np.asarray(img)
            img = img_to_array(img)
            train_data.append(img)
            #label = load_img(filepath + '\label\\' + url, grayscale=True)
            label = Image.open(filepath + '\label\\' + url).convert('1')
            label = img_to_array(label)
            #label = np.asarray(label)
            train_label.append(label)
            if batch % batch_size == 0:
                # print 'get enough bacth!\n'
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0

            # data for validation


def generateValidData(batch_size, data=[]):
    # print 'generateValidData...'
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            #img = load_img(filepath + '\src\\' + url)
            img=Image.open(filepath + '\src\\' + url).convert('RGB')
            img = img_to_array(img)
            valid_data.append(img)
            #label = load_img(filepath + '\label\\' + url, grayscale=True)
            label = Image.open(filepath + '\label\\' + url).convert('1')
            label = img_to_array(label)
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                batch = 0


def unet():
    concat_axis = 3
    inputs = Input((img_w, img_h, 3))
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same",data_format="channels_last")(inputs)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same",data_format="channels_last")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same",data_format="channels_last")(pool1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same",data_format="channels_last")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),data_format="channels_last")(conv2)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same",data_format="channels_last")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same",data_format="channels_last")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),data_format="channels_last")(conv3)

    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same",data_format="channels_last")(pool3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same",data_format="channels_last")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),data_format="channels_last")(conv4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same",data_format="channels_last")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same",data_format="channels_last")(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=concat_axis)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same",data_format="channels_last")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same",data_format="channels_last")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=concat_axis)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same",data_format="channels_last")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same",data_format="channels_last")(conv7)
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=concat_axis)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same",data_format="channels_last")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same",data_format="channels_last")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=concat_axis)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same",data_format="channels_last")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same",data_format="channels_last")(conv9)

    conv10 = Conv2D(n_label, (1, 1), activation="sigmoid",data_format="channels_last")(conv9)
    # conv10 = Conv2D(n_label, (1, 1), activation="softmax")(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr = 1e-6), loss=dice_coef_loss, metrics=['accuracy'])
    return model


def train(args):
    EPOCHS = 30
    BS = 5
    # model = SegNet()
    model = unet()
    modelcheck = ModelCheckpoint(args['model'], monitor='val_acc', save_best_only=True, mode='max')
    callable = [modelcheck]
    train_set, val_set = get_train_val()
    train_numb = len(train_set)
    valid_numb = len(val_set)
    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)
    H = model.fit_generator(generator=generateData(BS, train_set), steps_per_epoch=train_numb // BS, epochs=EPOCHS,verbose=1,validation_data=generateValidData(BS, val_set), validation_steps=valid_numb // BS,callbacks=callable, max_q_size=1)
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on U-Net Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", help="training data's path",
                    default=True)
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    args = args_parse()
    filepath = args['data']
    args['model']="unet_buildings22.h5"
    train(args)
    # predict()
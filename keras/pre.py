import cv2
import random
from PIL import Image
import keras
import numpy as np
import os
import argparse
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
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
from sklearn.preprocessing import LabelEncoder
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
from train import smooth

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TEST_SET = ['8 (1).png']

image_size = 256

classes = [0., 1.]

labelencoder = LabelEncoder()
labelencoder.fit(classes)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)
def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to trained model model")
    ap.add_argument("-s", "--stride", required=False,
                    help="crop slide stride", type=int, default=image_size)
    args = vars(ap.parse_args())
    return args


def predict(args):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model=args['model']
    weight_path = 'unet_buildings22.h5'
    model = load_model(weight_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    stride = args['stride']
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        # load the image
        image = Image.open('F:\新建文件夹 - 副本 (2)\赛道B附件\data\data\\' + path).convert('RGB')
        image = img_to_array(image)
        h, w, _ = image.shape
        padding_h = (h // stride + 1) * stride
        padding_w = (w // stride + 1) * stride
        padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
        padding_img[0:h, 0:w, :] = image[:, :, :]
        padding_img = padding_img.astype("float") / 255.0
        padding_img = img_to_array(padding_img)
        mask_whole = np.zeros((padding_h, padding_w), dtype=np.uint8)
        print(1)
        for i in range(padding_h // stride):
            print(2)
            for j in range(padding_w // stride):
                print(3)
                crop = padding_img[j * stride:j * stride + image_size, i * stride:i * stride + image_size, :3]
                ch, cw,_ = crop.shape
                crop = np.expand_dims(crop, axis=0)
                print("The crop shape:",crop.shape)
                pred = model.predict(crop, verbose=2)
                print('pred:', pred.shape)
                # print (np.unique(pred))
                pred = pred.reshape((256, 256)).astype(np.uint8)
                mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size] = pred[:, :]
        print('F:\新建文件夹 - 副本 (2)\赛道B附件\data\pre\mas\\' + str(n + 1) + '.png')
        mask_whole[0:h, 0:w] = mask_whole[0:h, 0:w] * 20
        cv2.imwrite(str(n + 1) + '.png', mask_whole[0:h, 0:w])


if __name__ == '__main__':
    args = args_parse()
    predict(args)

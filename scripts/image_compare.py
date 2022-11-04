import numpy as np
import argparse
import os
import cv2
from PIL import Image
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt


def concat_h(A, B):
    res = Image.new('RGB', (A.width + B.width, A.height))
    res.paste(A, (0, 0))
    res.paste(B, (A.width, 0))
    return res


def similarity(A, B):
    return np.sum(A == B)


def preprocess(img):
    img = np.array(img)
    img[img < np.mean(img)] = 0
    img[img >= np.mean(img)] = 255
    return Image.fromarray(img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training', type=str, help='Global path to training directory')
    parser.add_argument('images', type=str, help='Global path to image file (npz)')
    args = parser.parse_args()

    IMGS = np.load(args.images)
    imgs = IMGS['arr_0']
    n, h, w, ch = imgs.shape
    for i in range(n):
        img_sample = Image.fromarray(imgs[i, :, :, :]).convert('L')
        img_sample = img_sample.resize((128, 128))
        img_sample = preprocess(img_sample)
        np_sample = np.array(img_sample)

        img_closest = None, None
        sim_closest = -np.inf
        for path_train in os.listdir(args.training):
            path_train = os.path.join(args.training, path_train)
            img_train = Image.open(path_train).convert('L')
            img_train = img_train.resize((128, 128))
            np_train = np.array(img_train)

            s = similarity(np_train, np_sample)
            if s > sim_closest:
                img_closest = img_train
                sim_closest = s

        print("similarity: ", 100*sim_closest/(128*128), '%')
        concatenated = concat_h(img_sample, img_closest)
        concatenated.show()

        input("Press enter to process next image")

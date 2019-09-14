import os
import matplotlib.pyplot as plt
import numpy as np
import imageio
import glob
from PIL import Image
import csv
import pprint
import matplotlib.patches as patches
import cv2
import pandas as pd
import random



import time
import sys
import itertools
import math
import logging
import json
import re
from collections import OrderedDict
import matplotlib
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import skimage.draw
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("github/Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import model_cog
import imgaug
from imgaug import augmenters as iaa

from collections import Counter
# %matplotlib inline
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
print(MODEL_DIR)

FIRST_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
print(FIRST_WEIGHTS_PATH)
os.path.abspath(FIRST_WEIGHTS_PATH)
# data_path = os.path.join('/home/cognitive-comp/Рабочий стол/icevision', 'icevision_data/')








def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax







if __name__ == '__main__':


    config = model_cog.BalloonConfig()

    config.display()

    DEVICE = "/gpu:0"

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)

    new = 1
    if new == 1:
        weights_path = FIRST_WEIGHTS_PATH
        print("Loading weights ", weights_path)
        model.load_weights(weights_path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        weights_path = model.find_last()
        print("Loading weights ", weights_path)
        model.load_weights(weights_path, by_name=True)

    start_time = time.time()
    dataset_train = model_cog.BalloonDataset()
    data_path = '/dataset/training'
    ann_path = '/data/annotations'
    dataset_train.load_dataset(data_path,ann_path, "training")
    dataset_train.prepare()
    print(time.time() - start_time)
    print("Image Count: {}".format(len(dataset_train.image_ids)))

    start_time = time.time()
    dataset_val = model_cog.BalloonDataset()
    dataset_val.load_dataset(data_path,ann_path, "val")
    dataset_val.prepare()
    print(time.time() - start_time)
    print("Image Count: {}".format(len(dataset_val.image_ids)))

    augmentation = iaa.OneOf([
        #     iaa.Affine(scale = (0.5,2.0)),
        #     iaa.Affine(shear = (-10,10)),
        #     iaa.Affine(rotate= (-15,15)),
        iaa.Multiply(mul=(-0.4, 2.0)),
        iaa.AddToHueAndSaturation(value=(-100, 100)),
        iaa.GaussianBlur(sigma=(1, 3)),
        iaa.GammaContrast(gamma=(0.5, 3))
    ])


    def train(model, num_of_10):
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=10 * num_of_10,
                    layers='5+', augmentation=augmentation)


    num_epoch = 1
    for i in range(num_epoch):
        train(model, i)

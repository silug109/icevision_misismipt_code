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
from tqdm import tqdm
from collections import OrderedDict
import matplotlib
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import skimage.draw
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd;
import numpy as np;
from torch.utils.data import Dataset, DataLoader
import random;
import math;
from torchvision.datasets import FashionMNIST
from torch. utils.data import DataLoader
from matplotlib import pyplot as plt
import os
import cv2
import torch.nn.functional as F
import split_folders

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
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
print(MODEL_DIR)

FIRST_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
print(FIRST_WEIGHTS_PATH)
os.path.abspath(FIRST_WEIGHTS_PATH)
data_path = os.path.join('/home/cognitive-comp/Рабочий стол/icevision', 'icevision_data/')
name_classes_union = ['1.1', '1.11.1', '1.11.2', '1.12.1', '1.12.2', '1.13', '1.15', '1.16', '1.17', '1.20.1', '1.20.2', '1.20.3', '1.22', '1.23', '1.25', '1.3.1','1.31', '1.33', '1.34.1', '1.34.2', '1.34.3', '1.8', '2.1', '2.2', '2.3.1', '2.3.2', '2.4', '2.5', '3.1', '3.10', '3.11', '3.13', '3.18.1', '3.18.2', '3.19', '3.2', '3.20', '3.24', '3.25', '3.27', '3.28', '3.3', '3.31', '3.32', '3.4', '3.5','4.1.1', '4.1.2', '4.1.3', '4.1.4', '4.1.5', '4.1.6', '4.2.1', '4.2.2', '4.2.3', '4.3','4.4.1', '4.4.2', '4.5.1', '4.5.2', '5.14', '5.15.1', '5.15.2', '5.15.3', '5.15.4', '5.15.5', '5.15.6', '5.15.7', '5.16', '5.19.1', '5.19.2', '5.20', '5.21', '5.23.1', '5.24.1', '5.3', '5.31','5.32', '5.4', '5.5', '5.6', '5.7.1', '5.7.2', '6.10.1', '6.10.2', '6.11', '6.12', '6.13', '6.16', '6.18.3', '6.3.1', '6.4', '6.6', '6.7', '6.8.1', '6.9.1', '6.9.2', '7.19', '7.2', '7.3', '7.5', '8','8.1.1', '8.1.4', '8.11', '8.13', '8.14', '8.17', '8.2.1', '8.2.2', '8.2.3', '8.2.4', '8.2.5', '8.2.6', '8.21.1', '8.22.1', '8.22.2', '8.22.3', '8.23', '8.24', '8.3.1', '8.3.2', '8.4.1', '8.4.3', '8.5.2', '8.5.4', '8.6.1', '8.6.5', '8.7', '8.8', 'nan']
 
ind2labels_dict = {i+1:t for i,t in enumerate(name_classes_union)}

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def one_picture_detect(model_main, image_path = 'dog.png'):

    first_time = time.time()
    image = skimage.io.imread(image_path)
    print(image.shape)
    if image.shape[-1] == 4:
        image = image[..., :3]

    image, _, _, _, _ = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    # If grayscale. Convert to RGB for consistency.
    # image = np.concatenate((image, image[..., 0][..., np.newaxis]), axis=-1)

    print(image.shape)
    print(time.time() - first_time)
    # # Run object detection
    results = model_main.detect([image], verbose=0)
    print(time.time() - first_time)

    fig = plt.figure()
    plt.subplot(111)
    plt.imshow(image)
    plt.gca().add_patch(patches.Rectangle([1, 1], 10, 10))
    im = plt.gca()
    r = plt.gcf().canvas.get_renderer()
    plt.gca().draw(r)
    plt.savefig('suka.png')
    print(time.time() - first_time)

class Sign(object):
    def __init__(self,box, class_name):
        self.tracker = cv2.TrackerCSRT_create()
        self.box = box
        self.class_name = class_name
        self.counter = 0
    def get_counter(self):
        return self.counter
    def set_counter(self):
        self.counter += 1
    def set_tracker(self,frame):
        tracker = cv2.TrackerCSRT_create()
        ok = tracker.init(frame, self.box)
        self.tracker = tracker
    def update(self, frame):
        box = self.box
        if box[2] != 0 and box[3] != 0:
             ok, box = self.tracker.update(frame)
             self.box = box
    def get_box(self):
        return(self.box, self.class_name)

def isApear(box, image):
    x, y, w, h = box
    apear = False
    if x < 0 or y < 0 or x + w >= image.shape[1] or y + h >= image.shape[0]:
        apear = True
    return apear

def pnm_read(__full_path):
    im_rows = 2048
    im_cols = 2448
    im_size = im_rows * im_cols
    with open(__full_path) as raw_image:
        img = np.fromfile(raw_image, np.dtype(np.uint8), im_size).reshape((im_rows, im_cols))
        colour = cv2.demosaicing(img, cv2.COLOR_BAYER_GR2RGBA)
        colour = cv2.cvtColor(colour, cv2.COLOR_RGBA2BGR)
        # colour = cv2.resize(colour, (1280, 1070))
    return colour

def folder_evaluate(folder_path,subset, model):
    print('wearehere')
    file_tsv = open('file.tsv', 'w')
    file_tsv.write('frame\txtl\tytl\txbr\tybr\tclass\ttemporary\tdata\n')

    # file_tsv_2 = open('file.tsv', 'w')
    # file_tsv_2.write('frame\txtl\tytl\txbr\tybr\tclass\ttemporary\tdata\n')

    # final_data = []
    print(os.path.abspath(folder_path))
    data_path = '/dataset/training'
    counter = 0
    list_arr = os.listdir(os.path.join(data_path,folder_path,subset))
    list_arr.sort()
    counter = 0 
    signs = []
    use_track = 0
    for image_file in tqdm(list_arr[:600]):
        image_path = os.path.join(data_path,folder_path,subset,image_file)
        print(image_path)
        image = pnm_read(image_path)
        if use_track:

            counter += 1
            if counter % 30 == 0:
                cv2.imwrite('./picture/'+image_file[:-4]+'.jpg', image)
            if counter % 5 == 1:
                signs = []
                results = model.detect([image],verbose=0)
                r = results[0]
                print(len(r['rois']))
            #if counter%30 == 0:
             #   print('saved', './pictures',image_file, type(image), image.shape)
              #  cv2.imwrite('./pictures/'+image_file[:-4]+'.jpg', image)
                for i in range(len(r['class_ids'])):
                    coord = r['rois'][i]
                    class_inst = r['class_ids'][i]
                    if coord[2] - coord[0] < 5 or coord[3] - coord[1] < 5:
                        continue
                    if isApear((coord[0],coord[1],coord[2]-coord[0],coord[3]-coord[1]),image):continue
                    sign = Sign((coord[0],coord[1],coord[2]-coord[0], coord[3]-coord[1]),class_inst)
                    print(sign.get_box())
                    sign.set_tracker(image)
                    signs.append(sign)
                    #class_inst = r['class_ids'][i]
                    file_tsv.write(folder_path+'_'+subset+'/'+image_file[:-4] + '\t' + str(coord[1]) + '\t' + str(coord[0]) + '\t' + str(coord[3]) + '\t' + str(
                    coord[2]) + '\t' + ind2labels_dict[class_inst] + '\t'+''+'\t'+''+ '\n')
            else:
            # print('я долбаеб')
                for sign in signs:
                    box,class_name = sign.get_box()
                    x,y,w,h = box
                    if isApear(box, image):
                        signs.remove(sign)
                        continue
                    if w < 5 or h < 5:
                        signs.remove(sign)
                        continue
                    sign.update(image)
                    box ,class_name = sign.get_box()
                    x,y,w,h = box
                    xt = x
                    yt = y
                    xb = x + w
                    yb = y + h
                    file_tsv.write(folder_path+'_'+subset+'/'+ image_file[:-4]+ '\t' + str(yt)+ '\t'+ str(xt) + '\t'+ str(yb)+ '\t' + str(xb)+'\t'+ind2labels_dict[class_name]+'\t'+''+'\t'+''+'\n')
        else:
            if counter % 30 == 0:
                cv2.imwrite('./picture/' + image_file[:-4] + '.jpg', image)
            results = model.detect([image], verbose=0)
            r = results[0]
            print(len(r['rois']))
            for i in range(len(r['class_ids'])):
                coord = r['rois'][i]
                class_inst = r['class_ids'][i]
                file_tsv.write(
                    folder_path + '_' + subset + '/' + image_file[:-4] + '\t' + str(coord[1]) + '\t' + str(
                        coord[0]) + '\t' + str(coord[3]) + '\t' + str(
                        coord[2]) + '\t' + ind2labels_dict[class_inst] + '\t' + '' + '\t' + '' + '\n')

            # image_patch = image[coord[0] - 3:coord[2] + 3, coord[1] - 3:coord[3] + 3]
            # image_patch = cv2.resize(image_patch, (96, 96))
            # plt.figure()
            # plt.imshow(image_patch)
            # plt.show()
            # image_patch = np.swapaxes(image_patch, 0, 2)
            # sign_tensor = torch.tensor([image_patch], dtype=torch.float32)
            # temp_result = model_temp(sign_tensor)
            # temp = bool(temp_result.max(1)[1].data.item())
            #
            #
            # file_tsv_2.write(name_jpg + '\t' + str(coord[0]) + '\t' + str(coord[1]) + '\t' + str(coord[2]) + '\t' + str(
            #     coord[3]) + '\t' + dataset.ind2labels_dict[class_inst] + '\t' + str(temp) + '\t' + 'NA' + '\n')

    file_tsv.close()

if __name__ == '__main__':
    config = model_cog.BalloonConfig()
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1


    config = InferenceConfig()
    config.display()
    DEVICE = "/gpu:0"


    # class Model_speed(nn.Module):
    #     def __init__(self):
    #         super(Model_speed, self).__init__()
    #         self.layer1 = nn.Sequential(
    #             nn.Conv2d(3, 32, kernel_size=3, padding=1),
    #             nn.ReLU(),
    #             nn.BatchNorm2d(32),
    #             nn.MaxPool2d(2),
    #             nn.Conv2d(32, 32, kernel_size=5, padding=2),
    #             nn.ReLU(),
    #             nn.BatchNorm2d(32),
    #             nn.MaxPool2d(2),
    #             nn.Dropout(0.25))
    #         self.layer2 = nn.Sequential(
    #             nn.Conv2d(32, 64, kernel_size=5, padding=2),
    #             nn.ReLU(),
    #             nn.BatchNorm2d(64),
    #             nn.MaxPool2d(2),
    #             nn.Conv2d(64, 128, kernel_size=5, padding=2),
    #             nn.ReLU(),
    #             nn.BatchNorm2d(128),
    #             nn.MaxPool2d(2),
    #             nn.Dropout(0.25))
    #         self.fc = nn.Sequential(
    #             nn.Linear(2048, 20),
    #             nn.Sigmoid(),
    #             nn.Linear(20, 7),
    #             nn.Sigmoid())
    #
    #     def forward(self, x):
    #         out = self.layer1(x)
    #         out = self.layer2(out)
    #         out = out.view(out.size(0), -1)
    #         out = self.fc(out)
    #         return out
    #
    #
    # model_cars = Model_speed()
    # checkpoint = torch.load('model_speed_signs_pth.tar')
    # model_cars.load_state_dict(checkpoint['state_dict'])
    #
    #
    # class Model_temp(nn.Module):
    #     def __init__(self):
    #         super(Model_temp, self).__init__()
    #         self.layer1 = nn.Sequential(
    #             nn.Conv2d(3, 32, kernel_size=3, padding=1),
    #             nn.ReLU(),
    #             nn.BatchNorm2d(32),
    #             nn.MaxPool2d(2),
    #             nn.Conv2d(32, 32, kernel_size=5, padding=2),
    #             nn.ReLU(),
    #             nn.BatchNorm2d(32),
    #             nn.MaxPool2d(2),
    #             nn.Dropout(0.25))
    #         self.layer2 = nn.Sequential(
    #             nn.Conv2d(32, 64, kernel_size=5, padding=2),
    #             nn.ReLU(),
    #             nn.BatchNorm2d(64),
    #             nn.MaxPool2d(2),
    #             nn.Conv2d(64, 128, kernel_size=5, padding=2),
    #             nn.ReLU(),
    #             nn.BatchNorm2d(128),
    #             nn.MaxPool2d(2),
    #             nn.Dropout(0.25))
    #         self.fc = nn.Sequential(
    #             nn.Linear(4608, 2),
    #             nn.Sigmoid())
    #
    #     def forward(self, x):
    #         out = self.layer1(x)
    #         out = self.layer2(out)
    #         out = out.view(out.size(0), -1)
    #         out = self.fc(out)
    #         return out
    #
    #
    # model_temp = Model_temp()
    # checkpoint = torch.load('model_signs_temprary_pth.tar')
    # model_temp.load_state_dict(checkpoint['state_dict'])


    with tf.device(DEVICE):
        model_main = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # weights_path = FIRST_WEIGHTS_PATH
    # print("Loading weights ", weights_path)
    # model_main.load_weights(weights_path, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])


    weights_path = model_main.find_last()
    print("Loading weights ", weights_path)
    model_main.load_weights(weights_path, by_name=True)

    one_picture = 0
    if one_picture == 1:
        one_picture_detect(model_main)
    else:

        start_time = time.time()
        #folder_evaluate('2018-02-13_1523','left',model_main)
        folder_evaluate('2018-02-13_1418','left', model_main)
        # dataset_validate = model_cog.BalloonDataset()
        # dataset_validate.load_dataset_image_only(data_path)
        # dataset_validate.prepare()
        print('ШЛО ВСЁ:',time.time()- start_time)
        # print("Image Count: {}".format(len(dataset_validate.image_ids)))






    # def file_labels_create(model, dataset):
    #     file_tsv = open('file_mine_one_more.tsv', 'a')
    #     file_tsv.write('frame\txtl\tytl\txbr\tybr\tclass\n')
    #     #     print('frame\txtl\tytl\txbr\tybr\tclass\n')
    #     counter = 0
    #     for image_id in dataset.image_ids[5000:len(dataset.image_ids)]:
    #         #     for image_id in dataset.image_ids[0:10]:
    #
    #         counter += 1
    #         if counter % 20 == 0:
    #             print(counter)
    #         #'Change here'
    #         name_jpg = dataset.image_info[image_id]['path'].lstrip(
    #             '/home/cognitive-comp/Рабочий стол/icevision/icevision_data/jpgfiles_val/')[:-4]
    #         image = dataset.load_image(image_id)
    #         results = model_main.detect([image], verbose=0)
    #         r = results[0]
    #         for i in range(len(r['class_ids'])):
    #             coord = r['rois'][i]
    #             class_inst = r['class_ids'][i]
    #             #             print(name_jpg+'\t'+str(coord[0])+'\t'+str(coord[1])+'\t'+ str(coord[2]) + '\t' + str(coord[3])+ '\t'+ dataset.ind2labels_dict[class_inst]+'\n')
    #             file_tsv.write(
    #                 name_jpg + '\t' + str(coord[0]) + '\t' + str(coord[1]) + '\t' + str(coord[2]) + '\t' + str(
    #                     coord[3]) + '\t' + dataset.ind2labels_dict[class_inst] + '\n')
    #     file_tsv.close()

    # file_labels_create(model_main, dataset_validate)






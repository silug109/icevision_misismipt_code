""" MIPT

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import pandas as pd

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

def pnm_read(__full_path):
    im_rows = 2048
    im_cols = 2448
    im_size = im_rows * im_cols
    with open(__full_path) as raw_image:
        img = np.fromfile(raw_image, np.dtype(np.uint8), im_size).reshape((im_rows, im_cols))
        colour = cv2.demosaicing(img, cv2.COLOR_BAYER_GR2RGBA)
        colour = cv2.cvtColor(colour, cv2.COLOR_RGBA2RGB)
        # colour = cv2.resize(colour, (1280, 1070))
    return colour

class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "balloon"

    # We use a GPU with 12GB memory, which can fit two images.
    # Addataset_label_pathjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 131

    NUM_META_INF = 0
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


    IMAGE_CHANNEL_COUNT = 3

    MEAN_PIXEL = 3

    IMAGE_MIN_DIM = 2048
    IMAGE_MAX_DIM = 2560

    # IMAGE_RESIZE_MODE = 'NONE'




############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):

    # def load_balloon(self, data_path,ann_path,dataset_label_path,jpg_path, dataset_dir_arr, subset):
    def load_balloon(self, directory_of_data, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        # self.add_class("balloon", 1, "2.1")
        # self.add_class("balloon", 2, "2.4")
        # self.add_class("balloon", 3, "3.1")
        # self.add_class("balloon", 4, "3.24")
        # self.add_class("balloon", 5, "3.27")
        # self.add_class("balloon", 6, "4.1")
        # self.add_class("balloon", 7, "4.2")
        # self.add_class("balloon", 8, "5.19")
        # self.add_class("balloon", 9, "5.20")
        # self.add_class("balloon", 10, "8.22")

        # Train or validation dataset?
        assert subset in ["training", "val",'final']

        # name_classes_union = ['2.1','2.4','3.1','3.24','3.27','4.1','4.2','5.19','5.20','8.22']
        # name_classes_union = ['1.1', '1.11.1', '1.11.2', '1.12.1', '1.12.2', '1.13', '1.15', '1.16',
        #                       '1.17', '1.20.1', '1.20.2', '1.20.3', '1.22', '1.23', '1.25', '1.3.1',
        #                       '1.31', '1.33', '1.34.1', '1.34.2', '1.34.3', '1.8', '2.1', '2.2', '2.3.1',
        #                       '2.3.2', '2.4', '2.5', '3.1', '3.10', '3.11', '3.13', '3.18.1', '3.18.2', '3.19',
        #                       '3.2', '3.20', '3.24', '3.25', '3.27', '3.28', '3.3', '3.31', '3.32', '3.4', '3.5',
        #                       '4.1.1', '4.1.2', '4.1.3', '4.1.4', '4.1.5', '4.1.6', '4.2.1', '4.2.2', '4.2.3', '4.3',
        #                       '4.4.1', '4.4.2', '4.5.1', '4.5.2', '5.14', '5.15.1', '5.15.2', '5.15.3', '5.15.4', '5.15.5',
        #                       '5.15.6', '5.15.7', '5.16', '5.19.1', '5.19.2', '5.20', '5.21', '5.23.1', '5.24.1', '5.3', '5.31',
        #                       '5.32', '5.4', '5.5', '5.6', '5.7.1', '5.7.2', '6.10.1', '6.10.2', '6.11', '6.12', '6.13', '6.16',
        #                       '6.18.3', '6.3.1', '6.4', '6.6', '6.7', '6.8.1', '6.9.1', '6.9.2', '7.19', '7.2', '7.3', '7.5', '8',
        #                       '8.1.1', '8.1.4', '8.11', '8.13', '8.14', '8.17', '8.2.1', '8.2.2', '8.2.3', '8.2.4', '8.2.5', '8.2.6',
        #                       '8.21.1', '8.22.1', '8.22.2', '8.22.3', '8.23', '8.24', '8.3.1', '8.3.2', '8.4.1', '8.4.3', '8.5.2', '8.5.4',
        #                       '8.6.1', '8.6.5', '8.7', '8.8', 'nan']

        name_classes_union = ['1.1', '1.10', '1.11', '1.11.1', '1.12', '1.12.2', '1.13', '1.14', '1.15', '1.16', '1.17', '1.18', '1.19',
         '1.2', '1.20', '1.20.2', '1.20.3', '1.21', '1.22', '1.23', '1.25', '1.26', '1.27', '1.30', '1.31', '1.33',
         '1.5', '1.6', '1.7', '1.8', '2.1', '2.2', '2.3', '2.3.2', '2.3.3', '2.3.4', '2.3.5', '2.3.6', '2.4', '2.5',
         '2.6', '2.7', '3.1', '3.10', '3.11', '3.12', '3.13', '3.14', '3.16', '3.18', '3.18.2', '3.19', '3.2', '3.20',
         '3.21', '3.24', '3.25', '3.27', '3.28', '3.29', '3.30', '3.31', '3.32', '3.33', '3.4', '3.6', '4.1.1', '4.1.2',
         '4.1.3', '4.1.4', '4.1.5', '4.1.6', '4.2.1', '4.2.2', '4.2.3', '4.3', '4.5', '4.8.2', '4.8.3', '5.11', '5.12',
         '5.14', '5.15.1', '5.15.2', '5.15.3', '5.15.5', '5.15.7', '5.16', '5.17', '5.18', '5.19.1', '5.20', '5.21',
         '5.22', '5.3', '5.4', '5.5', '5.6', '5.7.1', '5.7.2', '5.8', '6.15.1', '6.15.2', '6.15.3', '6.16', '6.2',
         '6.3.1', '6.4', '6.6', '6.7', '6.8.1', '6.8.2', '6.8.3', '7.1', '7.11', '7.12', '7.14', '7.15', '7.18', '7.2',
         '7.3', '7.4', '7.5', '7.6', '7.7', '8.1.1', '8.1.3', '8.1.4', '8.13', '8.13.1', '8.14', '8.15', '8.16', '8.17',
         '8.18', '8.2.1', '8.2.2', '8.2.3', '8.2.4', '8.23', '8.3.1', '8.3.2', '8.3.3', '8.4.1', '8.4.3', '8.4.4',
         '8.5.2', '8.5.4', '8.6.2', '8.6.4', '8.8', 'nan']

        # name_classes_union = ['1.1', '1.10', '1.11', '1.11.1', '1.11.2', '1.12', '1.12.1', '1.12.2', '1.13', '1.14',
        #                       '1.15', '1.16', '1.17', '1.18', '1.19', '1.2', '1.20', '1.20.1', '1.20.2', '1.20.3',
        #                       '1.21', '1.22', '1.23', '1.25', '1.26', '1.27', '1.3.1', '1.30', '1.31', '1.33', '1.34.1',
        #                       '1.34.2', '1.34.3', '1.5', '1.6', '1.7', '1.8', '2.1', '2.2', '2.3', '2.3.1', '2.3.2',
        #                       '2.3.3', '2.3.4', '2.3.5', '2.3.6', '2.4', '2.5', '2.6', '2.7', '3.1', '3.10', '3.11',
        #                       '3.12', '3.13', '3.14', '3.16', '3.18', '3.18.1', '3.18.2', '3.19', '3.2', '3.20', '3.21',
        #                       '3.24', '3.25', '3.27', '3.28', '3.29', '3.3', '3.30', '3.31', '3.32', '3.33', '3.4',
        #                       '3.5', '3.6', '4.1.1', '4.1.2', '4.1.3', '4.1.4', '4.1.5', '4.1.6', '4.2.1', '4.2.2',
        #                       '4.2.3', '4.3', '4.4.1', '4.4.2', '4.5', '4.5.1', '4.5.2', '4.8.2', '4.8.3', '5.11',
        #                       '5.12', '5.14', '5.15.1', '5.15.2', '5.15.3', '5.15.4', '5.15.5', '5.15.6', '5.15.7',
        #                       '5.16', '5.17', '5.18', '5.19.1', '5.19.2', '5.20', '5.21', '5.22', '5.23.1', '5.24.1',
        #                       '5.3', '5.31', '5.32', '5.4', '5.5', '5.6', '5.7.1', '5.7.2', '5.8', '6.10.1', '6.10.2',
        #                       '6.11', '6.12', '6.13', '6.15.1', '6.15.2', '6.15.3', '6.16', '6.18.3', '6.2', '6.3.1',
        #                       '6.4', '6.6', '6.7', '6.8.1', '6.8.2', '6.8.3', '6.9.1', '6.9.2', '7.1', '7.11', '7.12',
        #                       '7.14', '7.15', '7.18', '7.19', '7.2', '7.3', '7.4', '7.5', '7.6', '7.7', '8', '8.1.1',
        #                       '8.1.3', '8.1.4', '8.11', '8.13', '8.13.1', '8.14', '8.15', '8.16', '8.17', '8.18',
        #                       '8.2.1', '8.2.2', '8.2.3', '8.2.4', '8.2.5', '8.2.6', '8.21.1', '8.22.1', '8.22.2',
        #                       '8.22.3', '8.23', '8.24', '8.3.1', '8.3.2', '8.3.3', '8.4.1', '8.4.3', '8.4.4', '8.5.2',
        #                       '8.5.4', '8.6.1', '8.6.2', '8.6.4', '8.6.5', '8.7', '8.8', 'nan']

        self.labels2ind_dict = {t:i+1 for i,t in enumerate(name_classes_union)}
        self.ind2labels_dict = {i+1:t for i,t in enumerate(name_classes_union)}


        for ind,name_class in enumerate(name_classes_union):
            self.add_class('balloon', ind, name_class)

        for dataset in os.listdir(os.path.join(directory_of_data,'jpgfiles/')):
            if dataset in os.listdir(os.path.join(directory_of_data,'annotations/',subset)):
                print('есть', dataset)
                for index_frame in (os.listdir(os.path.join(directory_of_data, 'annotations/', subset, dataset))):
                        #         print(index_frame)

                    bbox_list_cord = []
                    df = pd.read_csv(os.path.join(directory_of_data, 'annotations', subset, dataset, index_frame),
                                         sep='\t', dtype='str')
                    # df = df.drop(['temporary', 'data', 'occluded'], axis=1)
                    df.fillna({'data': 0}, inplace=True)
                    df = df.astype('str')

                    df['occluded'].replace(['true', 'false'], [1, 0], inplace=True)
                    df['temporary'].replace(['true', 'false'], [1, 0], inplace=True)

                    for i in df.index:
                        # x = (int(df.iloc[i][1]) + int(df.iloc[i][3])) / 2
                        # y = (int(df.iloc[i][2]) + int(df.iloc[i][4])) / 2
                        # h = int(abs(int(df.iloc[i][3]) - int(df.iloc[i][1])))
                        # w = int(abs(int(df.iloc[i][4]) - int(df.iloc[i][2])))

                        atr_box = tuple([item for item in df.iloc[i]])
                            # print(atr_box)
                        bbox_list_cord.append(atr_box)
                    image_path = os.path.join(directory_of_data,'jpgfiles',dataset,index_frame[:-3]+'jpg')
                    image = skimage.io.imread(image_path)
                    height, width = image.shape[:2]
                    print(image.shape)

                    self.add_image(
                        "balloon",
                        image_id= os.path.join(dataset,index_frame[:-4]+'.jpg') ,  # use file name as a unique image id
                        path=image_path,
                        width=width, height=height,
                        polygons=bbox_list_cord)

                print('загрузил епта, ', dataset)
            else:
                print('нет лейблов')




    def load_dataset(self, directory_of_dataset,directory_ann, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        print(directory_of_dataset,directory_ann)






        assert subset in ["training", "val",'final']


        name_classes_union = ['1.1', '1.10', '1.11', '1.11.1', '1.11.2', '1.12', '1.12.1', '1.12.2', '1.13', '1.14',
                              '1.15', '1.16', '1.17', '1.18', '1.19', '1.2', '1.20', '1.20.1', '1.20.2', '1.20.3',
                              '1.21', '1.22', '1.23', '1.25', '1.26', '1.27', '1.3.1', '1.30', '1.31', '1.33', '1.34.1',
                              '1.34.2', '1.34.3', '1.5', '1.6', '1.7', '1.8', '2.1', '2.2', '2.3', '2.3.1', '2.3.2',
                              '2.3.3', '2.3.4', '2.3.5', '2.3.6', '2.4', '2.5', '2.6', '2.7', '3.1', '3.10', '3.11',
                              '3.12', '3.13', '3.14', '3.16', '3.18', '3.18.1', '3.18.2', '3.19', '3.2', '3.20', '3.21',
                              '3.24', '3.25', '3.27', '3.28', '3.29', '3.3', '3.30', '3.31', '3.32', '3.33', '3.4',
                              '3.5', '3.6', '4.1.1', '4.1.2', '4.1.3', '4.1.4', '4.1.5', '4.1.6', '4.2.1', '4.2.2',
                              '4.2.3', '4.3', '4.4.1', '4.4.2', '4.5', '4.5.1', '4.5.2', '4.8.2', '4.8.3', '5.11',
                              '5.12', '5.14', '5.15.1', '5.15.2', '5.15.3', '5.15.4', '5.15.5', '5.15.6', '5.15.7',
                              '5.16', '5.17', '5.18', '5.19.1', '5.19.2', '5.20', '5.21', '5.22', '5.23.1', '5.24.1',
                              '5.3', '5.31', '5.32', '5.4', '5.5', '5.6', '5.7.1', '5.7.2', '5.8', '6.10.1', '6.10.2',
                              '6.11', '6.12', '6.13', '6.15.1', '6.15.2', '6.15.3', '6.16', '6.18.3', '6.2', '6.3.1',
                              '6.4', '6.6', '6.7', '6.8.1', '6.8.2', '6.8.3', '6.9.1', '6.9.2', '7.1', '7.11', '7.12',
                              '7.14', '7.15', '7.18', '7.19', '7.2', '7.3', '7.4', '7.5', '7.6', '7.7', '8', '8.1.1',
                              '8.1.3', '8.1.4', '8.11', '8.13', '8.13.1', '8.14', '8.15', '8.16', '8.17', '8.18',
                              '8.2.1', '8.2.2', '8.2.3', '8.2.4', '8.2.5', '8.2.6', '8.21.1', '8.22.1', '8.22.2',
                              '8.22.3', '8.23', '8.24', '8.3.1', '8.3.2', '8.3.3', '8.4.1', '8.4.3', '8.4.4', '8.5.2',
                              '8.5.4', '8.6.1', '8.6.2', '8.6.4', '8.6.5', '8.7', '8.8', 'nan']

        self.labels2ind_dict = {t:i+1 for i,t in enumerate(name_classes_union)}
        self.ind2labels_dict = {i+1:t for i,t in enumerate(name_classes_union)}

        for ind,name_class in enumerate(name_classes_union):
            self.add_class('balloon', ind, name_class)

        print(os.path.join(directory_ann,subset))
        for dataset in os.listdir(os.path.join(directory_ann,subset)):

            dataset_arr = dataset.split('_')
            dataset_name = dataset_arr[0]+'_'+dataset_arr[1]
            dataset_side = dataset_arr[-1]
            print(dataset,dataset_name,dataset_side)

            for image_tsv in os.listdir(os.path.join(directory_ann,subset,dataset)):
                image_file = image_tsv[:-4]+'.pnm'
                #print(image_file)
                #print(os.path.join(directory_of_dataset,dataset_name,dataset_side))
                if image_file in os.listdir(os.path.join(directory_of_dataset,dataset_name,dataset_side)):
                    bbox_list_cord = []
                    df = pd.read_csv(os.path.join(directory_ann, subset, dataset, image_tsv), sep='\t', dtype='str')
                    # df = df.drop(['temporary', 'data', 'occluded'], axis=1)
                    # df.fillna({'data': 0}, inplace=True)
                    df = df.astype('str')

                    df['occluded'].replace(['true', 'false'], [1, 0], inplace=True)
                    df['temporary'].replace(['true', 'false'], [1, 0], inplace=True)

                    for i in df.index:
                        # x = (int(df.iloc[i][1]) + int(df.iloc[i][3])) / 2
                        # y = (int(df.iloc[i][2]) + int(df.iloc[i][4])) / 2
                        # h = int(abs(int(df.iloc[i][3]) - int(df.iloc[i][1])))
                        # w = int(abs(int(df.iloc[i][4]) - int(df.iloc[i][2])))

                        atr_box = tuple([item for item in df.iloc[i]])
                            # print(atr_box)
                        bbox_list_cord.append(atr_box)
                    image_path = os.path.join(directory_of_dataset,dataset_name,dataset_side,image_file)
                    image = pnm_read(image_path)
                    height,width =image.shape[:2]
                    # image = skimage.io.imread(image_path)
                    # height, width = image.shape[:2]
                    # height,width = 2048, 2448

                    self.add_image(
                        "balloon",
                        image_id= os.path.join(dataset_name,dataset_side,image_file) ,  # use file name as a unique image id
                        path=image_path,
                        width=width, height=height,
                        polygons=bbox_list_cord)

            print('загрузил епта, ', dataset)

    def load_dataset_image_only(self,directory_of_data):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # self.add_class("balloon", 1, "2.1")
        # self.add_class("balloon", 2, "2.4")
        # self.add_class("balloon", 3, "3.1")
        # self.add_class("balloon", 4, "3.24")
        # self.add_class("balloon", 5, "3.27")
        # self.add_class("balloon", 6, "4.1")
        # self.add_class("balloon", 7, "4.2")
        # self.add_class("balloon", 8, "5.19")
        # self.add_class("balloon", 9, "5.20")
        # self.add_class("balloon", 10, "8.22")

        # name_classes_union = ['2.1','2.4','3.1','3.24','3.27','4.1','4.2','5.19','5.20','8.22']
        # name_classes_union = ['1.1', '1.11.1', '1.11.2', '1.12.1', '1.12.2', '1.13', '1.15', '1.16',
        #                       '1.17', '1.20.1', '1.20.2', '1.20.3', '1.22', '1.23', '1.25', '1.3.1',
        #                       '1.31', '1.33', '1.34.1', '1.34.2', '1.34.3', '1.8', '2.1', '2.2', '2.3.1',
        #                       '2.3.2', '2.4', '2.5', '3.1', '3.10', '3.11', '3.13', '3.18.1', '3.18.2', '3.19',
        #                       '3.2', '3.20', '3.24', '3.25', '3.27', '3.28', '3.3', '3.31', '3.32', '3.4', '3.5',
        #                       '4.1.1', '4.1.2', '4.1.3', '4.1.4', '4.1.5', '4.1.6', '4.2.1', '4.2.2', '4.2.3', '4.3',
        #                       '4.4.1', '4.4.2', '4.5.1', '4.5.2', '5.14', '5.15.1', '5.15.2', '5.15.3', '5.15.4',
        #                       '5.15.5',
        #                       '5.15.6', '5.15.7', '5.16', '5.19.1', '5.19.2', '5.20', '5.21', '5.23.1', '5.24.1', '5.3',
        #                       '5.31',
        #                       '5.32', '5.4', '5.5', '5.6', '5.7.1', '5.7.2', '6.10.1', '6.10.2', '6.11', '6.12', '6.13',
        #                       '6.16',
        #                       '6.18.3', '6.3.1', '6.4', '6.6', '6.7', '6.8.1', '6.9.1', '6.9.2', '7.19', '7.2', '7.3',
        #                       '7.5', '8',
        #                       '8.1.1', '8.1.4', '8.11', '8.13', '8.14', '8.17', '8.2.1', '8.2.2', '8.2.3', '8.2.4',
        #                       '8.2.5', '8.2.6',
        #                       '8.21.1', '8.22.1', '8.22.2', '8.22.3', '8.23', '8.24', '8.3.1', '8.3.2', '8.4.1',
        #                       '8.4.3', '8.5.2', '8.5.4',
        #                       '8.6.1', '8.6.5', '8.7', '8.8', 'nan']

        # name_classes_union = ['1.1', '1.10', '1.11', '1.11.1', '1.12', '1.12.2', '1.13', '1.14', '1.15', '1.16', '1.17',
        #                       '1.18', '1.19',
        #                       '1.2', '1.20', '1.20.2', '1.20.3', '1.21', '1.22', '1.23', '1.25', '1.26', '1.27', '1.30',
        #                       '1.31', '1.33',
        #                       '1.5', '1.6', '1.7', '1.8', '2.1', '2.2', '2.3', '2.3.2', '2.3.3', '2.3.4', '2.3.5',
        #                       '2.3.6', '2.4', '2.5',
        #                       '2.6', '2.7', '3.1', '3.10', '3.11', '3.12', '3.13', '3.14', '3.16', '3.18', '3.18.2',
        #                       '3.19', '3.2', '3.20',
        #                       '3.21', '3.24', '3.25', '3.27', '3.28', '3.29', '3.30', '3.31', '3.32', '3.33', '3.4',
        #                       '3.6', '4.1.1', '4.1.2',
        #                       '4.1.3', '4.1.4', '4.1.5', '4.1.6', '4.2.1', '4.2.2', '4.2.3', '4.3', '4.5', '4.8.2',
        #                       '4.8.3', '5.11', '5.12',
        #                       '5.14', '5.15.1', '5.15.2', '5.15.3', '5.15.5', '5.15.7', '5.16', '5.17', '5.18',
        #                       '5.19.1', '5.20', '5.21',
        #                       '5.22', '5.3', '5.4', '5.5', '5.6', '5.7.1', '5.7.2', '5.8', '6.15.1', '6.15.2', '6.15.3',
        #                       '6.16', '6.2',
        #                       '6.3.1', '6.4', '6.6', '6.7', '6.8.1', '6.8.2', '6.8.3', '7.1', '7.11', '7.12', '7.14',
        #                       '7.15', '7.18', '7.2',
        #                       '7.3', '7.4', '7.5', '7.6', '7.7', '8.1.1', '8.1.3', '8.1.4', '8.13', '8.13.1', '8.14',
        #                       '8.15', '8.16', '8.17',
        #                       '8.18', '8.2.1', '8.2.2', '8.2.3', '8.2.4', '8.23', '8.3.1', '8.3.2', '8.3.3', '8.4.1',
        #                       '8.4.3', '8.4.4',
        #                       '8.5.2', '8.5.4', '8.6.2', '8.6.4', '8.8', 'nan']

        name_classes_union = ['1.1', '1.10', '1.11', '1.11.1', '1.11.2', '1.12', '1.12.1', '1.12.2', '1.13', '1.14',
                              '1.15', '1.16', '1.17', '1.18', '1.19', '1.2', '1.20', '1.20.1', '1.20.2', '1.20.3',
                              '1.21', '1.22', '1.23', '1.25', '1.26', '1.27', '1.3.1', '1.30', '1.31', '1.33', '1.34.1',
                              '1.34.2', '1.34.3', '1.5', '1.6', '1.7', '1.8', '2.1', '2.2', '2.3', '2.3.1', '2.3.2',
                              '2.3.3', '2.3.4', '2.3.5', '2.3.6', '2.4', '2.5', '2.6', '2.7', '3.1', '3.10', '3.11',
                              '3.12', '3.13', '3.14', '3.16', '3.18', '3.18.1', '3.18.2', '3.19', '3.2', '3.20', '3.21',
                              '3.24', '3.25', '3.27', '3.28', '3.29', '3.3', '3.30', '3.31', '3.32', '3.33', '3.4',
                              '3.5', '3.6', '4.1.1', '4.1.2', '4.1.3', '4.1.4', '4.1.5', '4.1.6', '4.2.1', '4.2.2',
                              '4.2.3', '4.3', '4.4.1', '4.4.2', '4.5', '4.5.1', '4.5.2', '4.8.2', '4.8.3', '5.11',
                              '5.12', '5.14', '5.15.1', '5.15.2', '5.15.3', '5.15.4', '5.15.5', '5.15.6', '5.15.7',
                              '5.16', '5.17', '5.18', '5.19.1', '5.19.2', '5.20', '5.21', '5.22', '5.23.1', '5.24.1',
                              '5.3', '5.31', '5.32', '5.4', '5.5', '5.6', '5.7.1', '5.7.2', '5.8', '6.10.1', '6.10.2',
                              '6.11', '6.12', '6.13', '6.15.1', '6.15.2', '6.15.3', '6.16', '6.18.3', '6.2', '6.3.1',
                              '6.4', '6.6', '6.7', '6.8.1', '6.8.2', '6.8.3', '6.9.1', '6.9.2', '7.1', '7.11', '7.12',
                              '7.14', '7.15', '7.18', '7.19', '7.2', '7.3', '7.4', '7.5', '7.6', '7.7', '8', '8.1.1',
                              '8.1.3', '8.1.4', '8.11', '8.13', '8.13.1', '8.14', '8.15', '8.16', '8.17', '8.18',
                              '8.2.1', '8.2.2', '8.2.3', '8.2.4', '8.2.5', '8.2.6', '8.21.1', '8.22.1', '8.22.2',
                              '8.22.3', '8.23', '8.24', '8.3.1', '8.3.2', '8.3.3', '8.4.1', '8.4.3', '8.4.4', '8.5.2',
                              '8.5.4', '8.6.1', '8.6.2', '8.6.4', '8.6.5', '8.7', '8.8', 'nan']

        self.labels2ind_dict = {t:i+1 for i,t in enumerate(name_classes_union)}
        self.ind2labels_dict = {i+1:t for i,t in enumerate(name_classes_union)}

        for ind,name_class in enumerate(name_classes_union):
            self.add_class('balloon', ind, name_class)

        for dataset in os.listdir(os.path.join(directory_of_data,'jpgfiles_val/')):
                for index_frame in (os.listdir(os.path.join(directory_of_data,'jpgfiles_val',dataset))):

                    image_path = os.path.join(directory_of_data,'jpgfiles_val',dataset,index_frame)
#                     print(image_path)
                    image = skimage.io.imread(image_path)
                    height, width = image.shape[:2]

                    self.add_image(
                        "balloon",
                        image_id= os.path.join(dataset,index_frame[:-4]+'.jpg') ,  # use file name as a unique image id
                        path=image_path,
                        width=width, height=height)

                print('загрузил епта, ', dataset)

    def load_dataset_rtsd(self, location_path = '../rtsd_datat/',part =1 ,step = 1):

        # name_classes_union = ['1.1', '1.11.1', '1.11.2', '1.12.1', '1.12.2', '1.13', '1.15', '1.16',
        #                       '1.17', '1.20.1', '1.20.2', '1.20.3', '1.22', '1.23', '1.25', '1.3.1',
        #                       '1.31', '1.33', '1.34.1', '1.34.2', '1.34.3', '1.8', '2.1', '2.2', '2.3.1',
        #                       '2.3.2', '2.4', '2.5', '3.1', '3.10', '3.11', '3.13', '3.18.1', '3.18.2', '3.19',
        #                       '3.2', '3.20', '3.24', '3.25', '3.27', '3.28', '3.3', '3.31', '3.32', '3.4', '3.5',
        #                       '4.1.1', '4.1.2', '4.1.3', '4.1.4', '4.1.5', '4.1.6', '4.2.1', '4.2.2', '4.2.3', '4.3',
        #                       '4.4.1', '4.4.2', '4.5.1', '4.5.2', '5.14', '5.15.1', '5.15.2', '5.15.3', '5.15.4',
        #                       '5.15.5',
        #                       '5.15.6', '5.15.7', '5.16', '5.19.1', '5.19.2', '5.20', '5.21', '5.23.1', '5.24.1', '5.3',
        #                       '5.31',
        #                       '5.32', '5.4', '5.5', '5.6', '5.7.1', '5.7.2', '6.10.1', '6.10.2', '6.11', '6.12', '6.13',
        #                       '6.16',
        #                       '6.18.3', '6.3.1', '6.4', '6.6', '6.7', '6.8.1', '6.9.1', '6.9.2', '7.19', '7.2', '7.3',
        #                       '7.5', '8',
        #                       '8.1.1', '8.1.4', '8.11', '8.13', '8.14', '8.17', '8.2.1', '8.2.2', '8.2.3', '8.2.4',
        #                       '8.2.5', '8.2.6',
        #                       '8.21.1', '8.22.1', '8.22.2', '8.22.3', '8.23', '8.24', '8.3.1', '8.3.2', '8.4.1',
        #                       '8.4.3', '8.5.2', '8.5.4',
        #                       '8.6.1', '8.6.5', '8.7', '8.8', 'nan']

        # name_classes_union = ['1.1', '1.10', '1.11', '1.11.1', '1.12', '1.12.2', '1.13', '1.14', '1.15', '1.16', '1.17',
        #                       '1.18', '1.19',
        #                       '1.2', '1.20', '1.20.2', '1.20.3', '1.21', '1.22', '1.23', '1.25', '1.26', '1.27', '1.30',
        #                       '1.31', '1.33',
        #                       '1.5', '1.6', '1.7', '1.8', '2.1', '2.2', '2.3', '2.3.2', '2.3.3', '2.3.4', '2.3.5',
        #                       '2.3.6', '2.4', '2.5',
        #                       '2.6', '2.7', '3.1', '3.10', '3.11', '3.12', '3.13', '3.14', '3.16', '3.18', '3.18.2',
        #                       '3.19', '3.2', '3.20',
        #                       '3.21', '3.24', '3.25', '3.27', '3.28', '3.29', '3.30', '3.31', '3.32', '3.33', '3.4',
        #                       '3.6', '4.1.1', '4.1.2',
        #                       '4.1.3', '4.1.4', '4.1.5', '4.1.6', '4.2.1', '4.2.2', '4.2.3', '4.3', '4.5', '4.8.2',
        #                       '4.8.3', '5.11', '5.12',
        #                       '5.14', '5.15.1', '5.15.2', '5.15.3', '5.15.5', '5.15.7', '5.16', '5.17', '5.18',
        #                       '5.19.1', '5.20', '5.21',
        #                       '5.22', '5.3', '5.4', '5.5', '5.6', '5.7.1', '5.7.2', '5.8', '6.15.1', '6.15.2', '6.15.3',
        #                       '6.16', '6.2',
        #                       '6.3.1', '6.4', '6.6', '6.7', '6.8.1', '6.8.2', '6.8.3', '7.1', '7.11', '7.12', '7.14',
        #                       '7.15', '7.18', '7.2',
        #                       '7.3', '7.4', '7.5', '7.6', '7.7', '8.1.1', '8.1.3', '8.1.4', '8.13', '8.13.1', '8.14',
        #                       '8.15', '8.16', '8.17',
        #                       '8.18', '8.2.1', '8.2.2', '8.2.3', '8.2.4', '8.23', '8.3.1', '8.3.2', '8.3.3', '8.4.1',
        #                       '8.4.3', '8.4.4',
        #                       '8.5.2', '8.5.4', '8.6.2', '8.6.4', '8.8', 'nan']

        name_classes_union = ['1.1', '1.10', '1.11', '1.11.1', '1.11.2', '1.12', '1.12.1', '1.12.2', '1.13', '1.14',
                              '1.15', '1.16', '1.17', '1.18', '1.19', '1.2', '1.20', '1.20.1', '1.20.2', '1.20.3',
                              '1.21', '1.22', '1.23', '1.25', '1.26', '1.27', '1.3.1', '1.30', '1.31', '1.33', '1.34.1',
                              '1.34.2', '1.34.3', '1.5', '1.6', '1.7', '1.8', '2.1', '2.2', '2.3', '2.3.1', '2.3.2',
                              '2.3.3', '2.3.4', '2.3.5', '2.3.6', '2.4', '2.5', '2.6', '2.7', '3.1', '3.10', '3.11',
                              '3.12', '3.13', '3.14', '3.16', '3.18', '3.18.1', '3.18.2', '3.19', '3.2', '3.20', '3.21',
                              '3.24', '3.25', '3.27', '3.28', '3.29', '3.3', '3.30', '3.31', '3.32', '3.33', '3.4',
                              '3.5', '3.6', '4.1.1', '4.1.2', '4.1.3', '4.1.4', '4.1.5', '4.1.6', '4.2.1', '4.2.2',
                              '4.2.3', '4.3', '4.4.1', '4.4.2', '4.5', '4.5.1', '4.5.2', '4.8.2', '4.8.3', '5.11',
                              '5.12', '5.14', '5.15.1', '5.15.2', '5.15.3', '5.15.4', '5.15.5', '5.15.6', '5.15.7',
                              '5.16', '5.17', '5.18', '5.19.1', '5.19.2', '5.20', '5.21', '5.22', '5.23.1', '5.24.1',
                              '5.3', '5.31', '5.32', '5.4', '5.5', '5.6', '5.7.1', '5.7.2', '5.8', '6.10.1', '6.10.2',
                              '6.11', '6.12', '6.13', '6.15.1', '6.15.2', '6.15.3', '6.16', '6.18.3', '6.2', '6.3.1',
                              '6.4', '6.6', '6.7', '6.8.1', '6.8.2', '6.8.3', '6.9.1', '6.9.2', '7.1', '7.11', '7.12',
                              '7.14', '7.15', '7.18', '7.19', '7.2', '7.3', '7.4', '7.5', '7.6', '7.7', '8', '8.1.1',
                              '8.1.3', '8.1.4', '8.11', '8.13', '8.13.1', '8.14', '8.15', '8.16', '8.17', '8.18',
                              '8.2.1', '8.2.2', '8.2.3', '8.2.4', '8.2.5', '8.2.6', '8.21.1', '8.22.1', '8.22.2',
                              '8.22.3', '8.23', '8.24', '8.3.1', '8.3.2', '8.3.3', '8.4.1', '8.4.3', '8.4.4', '8.5.2',
                              '8.5.4', '8.6.1', '8.6.2', '8.6.4', '8.6.5', '8.7', '8.8', 'nan']

        sign_exclusive_class = ['3.11', '3.12', '3.13', '3.14', '3.24', '3.16', '3.24', '3.25', '3.4', '4.1.2',
                                '5.15.2', '6.2']



        self.labels2ind_dict = {t: i + 1 for i, t in enumerate(name_classes_union)}
        self.ind2labels_dict = {i + 1: t for i, t in enumerate(name_classes_union)}

        for ind, name_class in enumerate(name_classes_union):
            self.add_class('balloon', ind, name_class)

        print(os.path.join(location_path, 'full-gt.csv'))
        dat = pd.read_csv(os.path.join(location_path, 'full-gt.csv'))
        boxes = []
        # location_path = '../rtsd_datat/'
        for i in dat.index[:int(len(dat.index) / part):int(step)]:
            #     print(dat.iloc[i])
            filename = dat.iloc[i]['filename']
            x_from = int(dat.iloc[i]['x_from'])
            y_from = int(dat.iloc[i]['y_from'])
            width = int(dat.iloc[i]['width'])
            height = int(dat.iloc[i]['height'])
            sign_class = dat.iloc[i]['sign_class'].replace('_', '.')

            for item in sign_exclusive_class:
                if sign.startswith(item):
                    sign = item

            # if sign_class.startswith('3.4'):
            #     sign_class = '3.4'
            # if sign_class.startswith('4.1.2'):
            #     sign_class = '4.1.2'
            # if sign_class.startswith('3.24'):
            #     sign_class = '3.24'
            # if sign_class.startswith('5.15.2'):
            #     sign_class = '5.15.2'
            # if sign_class.startswith('3.14'):
            #     sign_class = '3.14'
            # # print(sign_class)

            bbox = tuple([sign_class, x_from, y_from, x_from + width, y_from + height])
            boxes.append(bbox)

            if filename != dat.iloc[i + 1]['filename']:
                image_path = os.path.join(location_path, 'rtsd-frames', filename)
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                # print(filename, boxes)
                self.add_image(
                    "balloon",
                    image_id=os.path.join(filename),  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    polygons=boxes)
                boxes = []
        print('загрузил, епта RTSD')

    def load_mask(self, image_id):

        info = self.image_info[image_id]
        mask = np.zeros((info["height"], info["width"], len(info["polygons"])),dtype=np.uint8)
        class_arr = []
        if len(info["polygons"]) > 0:
            for i, p in enumerate(info["polygons"]):
                # rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                r1 = int(p[2])
                r2 = int(p[4])
                c1 = int(p[1])
                c2 = int(p[3])
                mask[r1:r2, c1:c2, i] = 1
                class_arr.append(self.labels2ind_dict[p[0]])

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        return mask.astype(np.bool), np.array(class_arr, dtype = np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BalloonDataset()
    dataset_val.load_balloon(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')





############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BalloonConfig()
    else:
        class InferenceConfig(BalloonConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

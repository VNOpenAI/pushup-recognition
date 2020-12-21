import cv2
import numpy as np
import tensorflow as tf
import random
import json
import math
import os
import shutil
from myutils import *


class DataSequence(tf.keras.utils.Sequence):
    """ DataSequence for training,predicting
    """
    """
    # Arguments
        image_folder: path of folder contatin images
        file_json: path of json file contain information for train,test,val dataset  (string)
        batch_size: specify batch size for trainning or predicting (int)
        aug : specify if using data augmentation (Boolean)
    #__len__ : return len of steps_per_epoch
    #__getitem__: return A tuple(data,labels) data.shape=(batch_size,input.shape)
    
    """

    def __init__(self, image_folder, file_json, batch_size, aug=False):
        self.image_folder = image_folder
        list_data = json.load(open(file_json, 'r'))
        self.batch_size = batch_size
        self.aug = aug
        self.list_data = []
        for index, i in enumerate(list_data):
            self.list_data.append(i)
            if i['is_pushing_up'] == 1 and aug:
                for _ in range(5):
                    self.list_data.append(i)
        random.shuffle(self.list_data)

    def __len__(self,):
        return math.ceil(len(self.list_data)/self.batch_size)

    def __getitem__(self, index):
        data = self.list_data[index*self.batch_size:(index+1)*self.batch_size]
        batch_data = []
        batch_labels = []
        for i in data:
            image = cv2.imread(self.image_folder+'/'+i['image'])
            if image is None or image.ndim != 3:
                continue
            labels = i['is_pushing_up']
            if self.aug:
                if labels == 0:
                    image = croprandom(image)
                image = seq.augment_image(image)
            image = padding(image)
            image = cv2.resize(image, (224, 224))
            batch_data.append(image)
            batch_labels.append(labels)
        batch_data = np.array(batch_data)
        batch_labels = np.array(batch_labels)
        batch_data = (batch_data - 127.0) / 255.
        return batch_data, batch_labels

    def on_epoch_end(self):
        pass

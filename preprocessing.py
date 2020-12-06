import numpy as np
import os,shutil,json,random 
import cv2
import tensorflow as tf 
import tensorflow.keras as keras
import imgaug as ia
import imgaug.augmenters as iaa
import math
tf.random.set_seed(1)
np.random.seed(1)
data_folder='../data'
images_folder='../data/image'


seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
   # iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.3,
        iaa.GaussianBlur(sigma=(0, 0.3))
    ),
    # Strengthen or weaken the contrast in each image.
  #  iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
  #      rotate=(-25, 25),
   #     shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

class DataSequence(tf.keras.utils.Sequence):
    """ DataSequence for training,predicting
    """    
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        file_json: path of json file contain information for train,test,val dataset  (string)
        batch_size: specify batch size for trainning or predicting (int)
        aug : specify if using data augmentation (Boolean)

    #__len__ : return len of steps_per_epoch
    #__getitem__: return A tuple(data,labels) data.shape=(batch_size,input.shape)
    
    """
    def __init__(self,file_json,batch_size=32,aug=True):
        self.batch_size=batch_size
        self.aug=aug
        self.data=[]
        list_dict_of_data=json.load(open(file_json,'r'))['labels']
        for i in list_dict_of_data:
            if i['is_pushing_up'] == 1 and self.aug == True:       #handle unbalance data by x5 data_label1
                for _ in range(5):
                    self.data.append(i)
            else:
                self.data.append(i)
        random.shuffle(self.data)

    def __len__(self):
        return math.ceil(len(self.data) /self.batch_size)    

    def __getitem__(self, index):
        data = self.data[index*self.batch_size : self.batch_size* (index+1)] #len(data)=32
        data_batch_numpy=[]
        labels_batch_numpy=[]
        for i in data:
            image_numpy=[]
            for j in i['name']:
                single_image=cv2.imread(images_folder+'/'+j)
                if single_image is None:
                    print('Unable to read this image: ',j)
                    single_iamge=np.zeros((224,224,3))                   
                single_image=cv2.resize(single_image,dsize=(224,224))
                image_numpy.append(single_image)
            image_numpy=np.stack(image_numpy,axis=2)  #image_numpy.shape= 224,224,48
            data_batch_numpy.append(image_numpy)
            labels_batch_numpy.append(i['is_pushing_up'])  
        data_batch_numpy=np.array(data_batch_numpy)    #data_batch_numpy.shape = 32,224,224,48
        labels_batch_numpy=np.array(labels_batch_numpy)
        if self.aug == True:
            data_batch_numpy=seq(images=data_batch_numpy)           #augment data
        data_batch_numpy = (data_batch_numpy - 127.0 ) /255.

        return data_batch_numpy,labels_batch_numpy

    def on_epoch_end(self):
        pass


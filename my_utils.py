import imgaug
import numpy as np
import cv2
import tensorflow as tf
import random
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


def croprandom(images):
    """[this function crop a random small area in image, for augmentation ]

    Args:
        images ([numpy-array]): [an image or batch of images you want to crop]

    Returns:
        [numpy-array]: [cropped images]
    """
    tile = random.uniform(0.05, 0.3), random.uniform(0.05, 0.3)
    image = images.copy()
    height, width = image.shape[0], image.shape[1]
    crop_shape = (int(height*tile[0]), int(width*tile[1]))
    y = random.randint(0, height-crop_shape[0])
    x = random.randint(0, width - crop_shape[1])
    image[y:y+crop_shape[0], x:x+crop_shape[1],
          :] = np.zeros(shape=(crop_shape[0], crop_shape[1], 3))
    return image


def padding(images):
    """ this function padding a rectangle image into square image before resize
    """
    """Argument:
        images : an image or batch of images you want to pad

    Returns:
        [numpy-array]: [padded images ]
    """

    image = images.copy()
    if image.ndim == 3:
        height, width = image.shape[0], image.shape[1]
        delta = height-width
        if delta > 0:
            pad1 = np.zeros(shape=(height, delta//2, 3))
            pad2 = np.zeros(shape=(height, delta-delta//2, 3))
            image = np.concatenate([pad1, image, pad2], axis=1)
        else:
            delta = -delta
            pad1 = np.zeros(shape=(delta//2, width, 3))
            pad2 = np.zeros(shape=(delta-delta//2, width, 3))
            image = np.concatenate([pad1, image, pad2], axis=0)
    else:
        batch_image = []
        for i in range(image.shape[0]):
            single_image = image[i]
            height, width = single_image.shape[0], single_image.shape[1]
            delta = height-width
            if delta > 0:
                pad1 = np.zeros(shape=(delta//2, width, 3))
                pad2 = np.zeros(shape=(delta-delta//2, width, 3))
                single_image = np.concatenate(
                    [pad1, single_image, pad2], axis=1)
            else:
                delta = -delta
                pad1 = np.zeros(shape=(delta//2, width, 3))
                pad2 = np.zeros(shape=(delta-delta//2, width, 3))
                single_image = np.concatenate(
                    [pad1, single_image, pad2], axis=0)
            batch_image.append(single_image)
        image = np.array(batch_image)
    return image.astype('uint8')


"""[An imgaug.augmenters.Sequental object for augmentation]
"""
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0, 0.1)),
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 3))
    ),
    iaa.LinearContrast((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)  # apply augmenters in random order


def plot_images(save2file=False, samples=16, step=0, images=[]):
    """[summary]

    Args:
        save2file (bool, optional): [description]. Defaults to False.
        samples (int, optional): [quantity of images for plotting]. Defaults to 16.
        step (int, optional): [step for printing]. Defaults to 0.
        images (list of numpy-array, optional): [images you want to visualize]. Defaults to [].
    """
    print("    ____________________ Step = %d ____________________" % step)
    plt.figure(figsize=(6, 6))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        image = images[i, :, :, :]
        image = np.reshape(image, [224, 224, 3])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()

    if save2file:
        plt.savefig(str(step) + "_d.png")
        plt.close('all')
    else:
        plt.show()


def recall_m(y_true, y_pred):
    """[metrics for trainning model]

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]

    Returns:
        [type]: [description]
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """[metrics for trainning model]

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]

    Returns:
        [type]: [description]
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    """[metrics for trainning model]

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]

    Returns:
        [type]: [description]
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


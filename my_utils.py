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


seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Crop(percent=(0, 0.1)),  # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 3))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
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
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)  # apply augmenters in random order


def visualize(batch_image):
    fig = plt.figure()
    for i in range(16):
        fig.add_subplot(4, 4, i)
        image = batch_image[i]
        image = image.astype('uint8')


def plot_images(save2file=False, samples=16, step=0, images=[]):

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
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import json
import math
import os
from model import *
from my_utils import *
from preprocessing import *
np.random.seed(1)
tf.random.set_seed(1)

train = DataSequence('./data/images',
                     './data/train.json', batch_size=16, aug=True)
val = DataSequence('./data/images',
                   './data/val.json', batch_size=16, aug=False)

model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy', recall_m, precision_m, f1_m])


mc = ModelCheckpoint(filepath=os.path.join(
    './checkpoint', "model_ep{epoch:03d}.h5"), save_weights_only=False, save_format="h5", verbose=1)
tb = TensorBoard(log_dir='./log', write_graph=True)
history = model.fit(train,
                    epochs=45,
                    validation_data=val,
                    verbose=1,
                    callbacks=[tb, mc],
                    )

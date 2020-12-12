import json
import numpy as np
import tensorflow as tf
import tensorflow.keras


def create_model():
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False)
    out = tf.keras.layers.GlobalMaxPooling2D()(mobilenet.layers[-1].output)
    out = tf.keras.layers.Dropout(0.5)(out)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(out)
    model = tf.keras.Model(mobilenet.layers[0].input, out)
    return model

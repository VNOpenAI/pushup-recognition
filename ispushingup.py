import numpy as np
import cv2 
import tensorflow as tf 
from my_utils import padding
class SimpleKalmanFilter():
    """[KalmanFilter using to eliminate gauss noise ]
    """
    def __init__(self, mea_e,  est_e,  q, current_estimate=0.5):
        self.err_measure = mea_e
        self.err_estimate = est_e
        self.q = q
        self.current_estimate = current_estimate
        self.last_estimate = current_estimate
        self.kalman_gain = 0

    def updateEstimate(self, mea):
        self.kalman_gain = self.err_estimate / \
            (self.err_estimate + self.err_measure)
        self.current_estimate = self.last_estimate + \
            self.kalman_gain * (mea - self.last_estimate)
        self.err_estimate = (1.0 - self.kalman_gain) * self.err_estimate + \
            abs(self.last_estimate-self.current_estimate)*self.q
        self.last_estimate = self.current_estimate

        return self.current_estimate


class IsPushingUp():
    def __init__(self,):
        self.model = tf.keras.models.load_model(
            'model/model_ep012.h5',compile =False)
        self.boloc = SimpleKalmanFilter(2, 2, 2, 0.5)
        self.last_score_raw = 0.5
        self.current_score_raw = 0
        self.current_score = 0

    def predict(self, frames):
        frame = frames.copy()
        frame = padding(frame)
        frame = cv2.resize(frame, dsize=(224, 224))
        frame = (frame - 127.0) / 255.
        frame = np.expand_dims(frame, axis=0)
        self.current_score_raw = self.model.predict(frame)
        self.current_score = self.boloc.updateEstimate(self.last_score_raw)
        self.last_score_raw = self.current_score_raw

        return self.current_score
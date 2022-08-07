# def weighted_categorical_crossentropy(weights):
#     # weights = [0.9,0.05,0.04,0.01]
#     def wcce(y_true, y_pred):
#         Kweights = K.constant(weights)
#         if not K.is_tensor(y_pred): y_pred = K.constant(y_pred)
#         y_true = K.cast(y_true, y_pred.dtype)
#         return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
#     return wcce

import tensorflow as tf
import  tensorflow.keras.backend as K

class weighted_loss(tf.keras.losses.Loss):
    def __init__(self,weights):
        super(weighted_loss,self).__init__()
        self.weights=weights
        
    def call(self,y_true:tf.float32,y_pred:tf.float32):
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * self.weights, axis=-1)
        
        
        
from tensorflow.python import keras
from itertools import product
import numpy as np
from tensorflow.python.keras.utils import losses_utils

class WeightedCategoricalCrossentropy(keras.losses.CategoricalCrossentropy):

    def __init__(
        self,
        weights,
        from_logits=False,
        label_smoothing=0,
        reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
        name='categorical_crossentropy',
    ):
        super().__init__(
            from_logits, label_smoothing, reduction, name=f"weighted_{name}"
        )
        self.weights = weights

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true,tf.float32)
        y_pred = tf.cast(y_pred,tf.float32)
        weights = self.weights
        nb_cl = len(weights)
        final_mask = keras.backend.zeros_like(y_pred[:, 0])
        y_pred_max = keras.backend.max(y_pred, axis=1)
        y_pred_max = keras.backend.reshape(
            y_pred_max, (keras.backend.shape(y_pred)[0], 1))
        y_pred_max_mat = keras.backend.cast(
            keras.backend.equal(y_pred, y_pred_max), keras.backend.floatx())
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (
                weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return super().call(y_true, y_pred) * final_mask
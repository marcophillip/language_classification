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
        
        
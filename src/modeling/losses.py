import tensorflow as tf
from tensorflow.keras import losses


def get_loss(loss_function):
    if loss_function == 'mse':
        return losses.MeanSquaredError()
    elif loss_function == 'mae':
        return losses.MeanAbsoluteError()
    elif loss_function == 'reconmse' or loss_function == 'recon_mse':
        return ReconMSE()


class ReconMSE(losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1) +\
    tf.reduce_mean(tf.math.square(tf.math.abs(y_pred) - tf.math.abs(y_true)), axis=-1) +\
    tf.reduce_mean(tf.math.square(tf.math.angle(y_pred) - tf.math.angle(y_true)), axis=-1)

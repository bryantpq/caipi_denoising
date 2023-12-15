import logging
import numpy as np
import os
import tensorflow as tf
import pdb

from tensorflow.keras import losses
from tensorflow.keras.models import Model

from modeling.models import dncnn, cdncnn, scnn, ddlr
from preparation.preprocessing_pipeline import fourier_transform as ft, inverse_fourier_transform as ift

def get_model(
        dimensions,
        model_type, 
        input_shape,
        load_model_path,
        loss_function='mse',
        learning_rate=0.001,
        window_size=7,
        init_alpha=None,
        noise_window_size=[32, 32]
    ):

    assert len(input_shape) == dimensions + 2, "Expected input_shape should be 2 more than the number of data dimensions (for num_samples and num_channels). Got {}".format(input_shape)
    
    img_size = input_shape[1:] # (384, 384, 1)
    
    logging.info(f'    Using {model_type}')
    if model_type == 'dncnn':
        model = dncnn(dimensions, input_shape=img_size)

    elif model_type == 'cdncnn':
        model = cdncnn(dimensions, input_shape=img_size)

    elif model_type == 'scnn':
        model = scnn(input_shape=img_size, init_alpha=init_alpha, noise_window_size=noise_window_size)

    elif model_type == 'ddlr':
        model = ddlr(input_shape=img_size, window_size=window_size)

    optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
    )

    model.build(input_shape=input_shape)
    model.compile(optimizer=optimizer,
                  loss=get_loss(loss_function), 
                  metrics=[]
    )
    
    if load_model_path is not None and load_model_path != '' :
        logging.info('    Loading model weights: {}'.format(load_model_path))
        model.load_weights(load_model_path)
    
    return model


def get_loss(loss_function):
    if loss_function == 'mse':
        return losses.MeanSquaredError()
    elif loss_function == 'mae':
        return losses.MeanAbsoluteError()
    elif loss_function == 'reconmse' or loss_function == 'recon_mse':
        return ReconMSE()


class ReconMSE(losses.Loss):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def call(self, y_true, y_pred):
        tmp_true, tmp_pred = tf.identity(y_true), tf.identity(y_pred)
        tf.print('here')
        tf.print(tmp_true.shape)

        tmp_true = tf.signal.ifftshift(tmp_true, axes=[1,2])
        tmp_pred = tf.signal.ifftshift(tmp_pred, axes=[1,2])

        # transpose dims for fft operation
        # y_true.shape = [None, 384, 384, 1] -> [None, 1, 384, 384]
        tmp_true = tf.cast(tf.transpose(tmp_true, perm=[0,3,1,2]), tf.complex64)
        tmp_pred = tf.cast(tf.transpose(tmp_pred, perm=[0,3,1,2]), tf.complex64)
        tf.print(tmp_true.shape)

        ift_true = tf.signal.ifft2d(tmp_true)
        ift_pred = tf.signal.ifft2d(tmp_pred)
        tf.print(ift_true.shape)

        ift_true = tf.transpose(ift_true, perm=[0,2,3,1])
        ift_pred = tf.transpose(ift_pred, perm=[0,2,3,1])
        tf.print(ift_true.shape)

        mag_true, mag_pred = tf.math.abs(ift_true), tf.math.abs(ift_pred)

        loss = tf.math.abs(tf.keras.losses.mean_squared_error(y_true, y_pred)) +\
               self.beta * tf.keras.losses.mean_squared_error(mag_true, mag_pred)

        return loss

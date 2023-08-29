import logging
import os
import tensorflow as tf
import pdb

from tensorflow.keras import losses
from tensorflow.keras.models import Model
from modeling.models import dncnn, cdncnn, scnn, ddlr

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
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1) +\
    tf.reduce_mean(tf.math.square(tf.math.abs(y_pred) - tf.math.abs(y_true)), axis=-1) +\
    tf.reduce_mean(tf.math.square(tf.math.angle(y_pred) - tf.math.angle(y_true)), axis=-1)

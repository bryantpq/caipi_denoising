import logging
import os
import tensorflow as tf

from tensorflow.keras import losses
from tensorflow.keras.models import Model
from modeling.models import dncnn, res_dncnn, complex_dncnn
from modeling.losses import get_loss


def get_model(
        model_type, 
        loss_function,
        input_shape=None, 
        load_model_path=None,
        learning_rate=0.001
    ):
    assert len(input_shape) == 4, "Expected input_shape to be 4-dim. Got {}".format(input_shape)
    
    img_size = input_shape[1:]
    
    logging.info(f'    Using {model_type}')
    if model_type == 'dncnn':
        model = dncnn(input_shape=img_size)

    elif model_type == 'res_dncnn':
        model = res_dncnn(input_shape=img_size)

    elif model_type == 'complex_dncnn':
        model = complex_dncnn(input_shape=img_size)

    model.build(input_shape=input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=get_loss(loss_function), 
                  metrics=[])
    
    if load_model_path is not None and load_model_path != '' :
        logging.info('    Loading model weights: {}'.format(load_model_path))
        model.load_weights(load_model_path)
    
    return model

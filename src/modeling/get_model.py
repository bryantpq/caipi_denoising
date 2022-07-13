import logging
import os
import tensorflow as tf

from tensorflow.keras import layers, losses, Sequential, Input
from tensorflow.keras.models import Model
from modeling.models import Denoiser, get_model1, get_model2, get_model3, get_model4


def get_model(model_type, input_shape=None, load_model_path=None):
    assert len(input_shape) == 4, "Expected input_shape to be 4-dim. Got {}".format(input_shape)
    
    img_size = input_shape[1:]
    
    if model_type == 0:
        logging.info('    Using model type 0')
        model = Denoiser(input_shape=img_size).model
        
    elif model_type == 1:
        logging.info('    Using model type 1')
        model = get_model1(input_shape=img_size)

    elif model_type == 2:
        logging.info('    Using model type 2')
        model = get_model2(input_shape=img_size)

    elif model_type == 3:
        logging.info('    Using model type 3')
        model = get_model3(input_shape=img_size)

    elif model_type == 4:
        logging.info('    Using model type 4')
        model = get_model4(input_shape=img_size)

    model.build(input_shape=input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=losses.MeanSquaredError(), 
                  metrics=[])
    
    if load_model_path is not None and load_model_path != '' :
        logging.info('    Loading model weights: {}'.format(load_model_path))
        model.load_weights(load_model_path)
    
    return model

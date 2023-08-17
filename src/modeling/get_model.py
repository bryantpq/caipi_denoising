import logging
import os
import tensorflow as tf

from tensorflow.keras import losses
from tensorflow.keras.models import Model
from modeling.models import dncnn, complex_dncnn, scnn, ddlr
from modeling.losses import get_loss

def get_model(
        model_type, 
        loss_function,
        input_shape=None, 
        load_model_path=None,
        learning_rate=0.001,
        window_size=7,
        init_alpha=None,
        noise_window_size=[32, 32]
    ):
    assert len(input_shape) == 4, "Expected input_shape to be 4-dim. Got {}".format(input_shape)
    
    img_size = input_shape[1:] # (384, 384, 1)
    
    logging.info(f'    Using {model_type}')
    if model_type == 'dncnn':
        model = dncnn(input_shape=img_size)

    elif model_type == 'complex_dncnn':
        model = complex_dncnn(input_shape=img_size)

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
                  metrics=[])
    
    if load_model_path is not None and load_model_path != '' :
        logging.info('    Loading model weights: {}'.format(load_model_path))
        model.load_weights(load_model_path)
    
    return model

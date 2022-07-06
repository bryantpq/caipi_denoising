import numpy as np
import tensorflow as tf


def preprocess_slices(data,
                      preprocessing_params,
                      steps):

    assert len(data) != 4, 'data should have 4 dimensions. Given {}'.format(data.shape)
    
    pipeline = gen_pipeline(steps=steps)
    for func, name in pipeline:
        print('   step: {}'.format(name))
        if preprocessing_params[name] is None:
            data = func(data)
        else:
            data = func(data, **preprocessing_params[name])
        
    return data


def gen_pipeline(steps):
    """
    Return list of functions to apply onto data.
    """
    pipeline = []
    
    if steps is not None:
        for step in steps:
            if   step == 'pad_square':
                pipeline.append( (pad_square, step) )

            elif step == 'normalize':
                pipeline.append( (normalize, step) )

            elif step == 'white_noise':
                pipeline.append( (white_noise, step) )

            elif step == 'random_xy_flip':
                pipeline.append( (random_xy_flip, step) )

            else:
                print('Operation {} not supported'.format(step))
    
    return pipeline


"""
Preprocessing Operations
"""

def normalize(X):
    mean = np.mean(X)
    std  = np.std(X)
    
    return (X - mean) / std


def pad_square(X, 
               pad_value=0.0):
    pad_len = (X.shape[1] - X.shape[2]) // 2
    X = np.pad(
        X, 
        [(0, 0), (0, 0), (pad_len, pad_len), (0, 0)],
        constant_values=pad_value
    )
    
    return X


def white_noise(X, 
                mu=0.0, 
                sigma=0.1):
    noise_map = np.random.normal(mu, sigma, (X.shape))
    X = X + noise_map

    return X
    

def random_xy_flip(X):
    for img_i in range(X.shape[0]):
        img = X[img_i]
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        X[img_i] = img
        
    return X

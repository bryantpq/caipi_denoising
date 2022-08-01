import logging
import numpy as np
import tensorflow as tf


def preprocess_slices(data,
                      preprocessing_params,
                      steps):

    assert len(data) != 4, 'data should have 4 dimensions. Given {}'.format(data.shape)
    
    data = data.astype('float32')
    pipeline = gen_pipeline(steps=steps)
    for func, name in pipeline:
        logging.info('   step: {}'.format(name))
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
            if step == 'normalize':
                pipeline.append( (normalize, step) )

            elif   step == 'pad_square':
                pipeline.append( (pad_square, step) )

            elif step == 'random_xy_flip':
                pipeline.append( (random_xy_flip, step) )

            elif step == 'standardize':
                pipeline.append( (standardize, step) )

            elif step == 'threshold_intensities':
                pipeline.append( (threshold_intensities, step) )

            elif step == 'white_noise':
                pipeline.append( (white_noise, step) )

            else:
                logging.info('Operation {} not supported'.format(step))
    
    return pipeline


"""
Preprocessing Operations
"""

def normalize(X,
              mode='batch'):
    if mode == 'batch':
        mean, std = np.mean(X), np.std(X)

        return (X - mean) / std

    elif mode == 'slice':
        for i in range(len(X)):
            mean, std = np.mean(X[i]), np.std(X[i])

            X[i] = (X[i] - mean) / std

        return X


def standardize(X,
                mode='subject'):
    if mode == 'subject':
        std_X = np.zeros(X.shape, dtype='float32')

        for subj_i in range(int(len(X) / 256)):
            subj_vol = X[subj_i * 256: subj_i * 256 + 256]
            min_val, max_val = np.min(subj_vol), np.max(subj_vol)
            num = subj_vol - min_val
            den = max_val - min_val
            std_X[subj_i * 256: subj_i * 256 + 256] = num / den

        return std_X

    elif mode == 'batch':
        min_ds, max_ds = np.min(X), np.max(X)

        num = X - min_ds
        den = max_ds - min_ds

        return num / den

    elif mode == 'slice':
        for i in range(len(X)):
            min_slc, max_slc = np.min(X[i]), np.max(X[i])

            num = X[i] - min_slc
            den = max_slc - min_slc

            X[i] = num / den

        return X


def threshold_intensities(X, 
                          value=5000):
    orig_len = len(X)
    sum_intensities = np.array([ np.sum(X[i]) for i in range(len(X)) ])
    X = X[np.where(sum_intensities > value)]
    new_len = len(X)
    logging.info(f'        Original n slices: {orig_len}')
    logging.info(f'        New n slices: {new_len}')

    return X


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
                sigma=0.2):
    for i in range(len(X)):
        max_val = np.max(X[i])
        noise_map = np.random.normal(mu, sigma * max_val, X[i].shape)
        X[i] = X[i] + noise_map

    return X
    

def random_xy_flip(X,
                   seed=24):
    for img_i in range(X.shape[0]):
        img = X[img_i]
        img = tf.image.stateless_random_flip_up_down(img, seed=[img_i, seed])
        img = tf.image.stateless_random_flip_left_right(img, seed=[img_i, seed * seed])
        X[img_i] = img
        
    return X

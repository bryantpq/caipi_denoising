import logging
import numpy as np
import tensorflow as tf


def preprocess_slices(data,
                      dimensions,
                      preprocessing_params,
                      steps):

    if dimensions == 2:
        assert data.ndim == 3, 'Expected 3 dimensions. Given {}'.format(data.shape)

        data = np.expand_dims(data, axis=3)
        logging.info('Adding dimension for 2D data, {}'.format(data.shape))

    elif dimensions == 3:
        assert data.ndim == 4, 'Expected 4 dimensions. Given {}'.format(data.shape)

    data = data.astype('float32')
    pipeline = gen_pipeline(steps=steps)
    for func, name in pipeline:
        logging.info('   step: {}'.format(name))
        if preprocessing_params[name] is None:
            data = func(data)
        else:
            preprocessing_params[name]['dimensions'] = dimensions
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

            elif step == 'pad_square':
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
              dimensions=3,
              mode='iterate'):
    '''
    Change the distribution of the dataset to have mean=0, std=1
    '''
    logging.debug(X.shape)
    if mode == 'batch':
        mean, std = np.mean(X), np.std(X)

        X = (X - mean) / std

    else:
        for i in range(len(X)):
            mean, std = np.mean(X[i]), np.std(X[i])

            X[i] = (X[i] - mean) / std

    logging.debug(X.shape)
    return X


def pad_square(X, 
               dimensions=3,
               pad_value=0.0):
    logging.debug(X.shape)
    if dimensions == 2:
        pad_len = (X.shape[1] - X.shape[2]) // 2
        X = np.pad(
                X, 
                [(0, 0), (0, 0), (pad_len, pad_len), (0, 0)],
                constant_values=pad_value
            )
    elif dimensions == 3:
        pad_len = [(0, 0)]
        largest_dim = np.max(X.shape[1:])
        for dim in X.shape[1:]:
            diff = largest_dim - dim
            pad_len.append( (diff // 2, diff // 2) )
        X = np.pad(X, pad_len, constant_values=pad_value)
    
    logging.debug(X.shape)

    return X
    

def random_xy_flip(X,
                   dimensions=3,
                   seed=24):
    logging.debug(X.shape)
    for img_i in range(X.shape[0]):
        img = X[img_i]
        img = tf.image.stateless_random_flip_up_down(img, seed=[img_i, seed])
        img = tf.image.stateless_random_flip_left_right(img, seed=[img_i, seed * seed])
        X[img_i] = img

    logging.debug(X.shape)
    return X


def standardize(X,
                dimensions=3,
                mode='subject'):
    logging.debug(X.shape)
    assert mode == 'subject', 'oops'

    if dimensions == 2:
        std_X = np.zeros(X.shape, dtype='float32')

        for subj_i in range(int(len(X) / 256)):
            subj_vol = X[subj_i * 256: subj_i * 256 + 256]
            min_val, max_val = np.min(subj_vol), np.max(subj_vol)
            num = subj_vol - min_val
            den = max_val - min_val
            std_X[subj_i * 256: subj_i * 256 + 256] = num / den

    elif dimensions == 3:
        std_X = np.zeros(X.shape, dtype='float32')

        for subj_i in range(len(X)):
            min_val, max_val = np.min(X[subj_i]), np.max(X[subj_i])
            num = X[subj_i] - min_val
            den = max_val - min_val
            std_X[subj_i] = num / den

    logging.debug(X.shape)
    return std_X


def threshold_intensities(X, 
                          dimensions=3,
                          value=5000):

    assert dimensions == 2, f'threshold_intensities not implemented for {dimensions} dimensions.'

    logging.debug(X.shape)
    orig_len = len(X)
    logging.info(f'    Before: {orig_len}')
    sum_intensities = np.array([ np.sum(X[i]) for i in range(len(X)) ])
    X = X[np.where(sum_intensities > value)]
    new_len = len(X)
    logging.info(f'    After: {new_len}')
    logging.debug(X.shape)

    return X


def white_noise(X, 
                dimensions=3,
                mu=0.0, 
                sigma=0.2):
    logging.debug(X.shape)
    for i in range(len(X)):
        max_val = np.max(X[i])

        if type(sigma) == list:
            cur_sigma = np.random.choice(sigma)
            noise_map = np.random.normal(mu, cur_sigma * max_val, X[i].shape)
        else:
            noise_map = np.random.normal(mu, sigma * max_val, X[i].shape)

        X[i] = X[i] + noise_map

    logging.debug(X.shape)
    return X

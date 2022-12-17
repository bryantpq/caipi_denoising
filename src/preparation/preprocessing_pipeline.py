import logging
import numpy as np
import tensorflow as tf
import pdb


def preprocess_data(data, params, steps):
    '''
    Given numpy array of data, return list of arrays for each subject
    '''
    assert data.ndim == 4, 'Expected 4 dimensions. Given {}'.format(data.shape)

    res = []

    if np.iscomplexobj(data):
        data = data.astype('complex64')
    else:
        data = data.astype('float32')

    pipeline = gen_pipeline(steps=steps)
    for subj_i, subject_data in enumerate(data):
        temp_data = np.copy(subject_data)
        pp_subj = temp_data
        for func, name in pipeline:
            if name == 'random_xy_flip':
                pp_subj = func(pp_subj, subj_i, **params[name])
            elif params[name] is None:
                pp_subj = func(pp_subj)
            else:
                pp_subj = func(pp_subj, **params[name])
        res.append(pp_subj)

    return res

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

def normalize(data):
    '''
    Change the distribution of the subject to have mean=0, std=1
    '''
    logging.debug(data.shape)

    mean, std = np.mean(data), np.std(data)
    data = (data - mean) / std

    logging.debug(data.shape)

    return data

def pad_square(data, pad_value=0.0):
    logging.debug(data.shape)

    pad_len = (data.shape[0] - data.shape[1]) // 2
    data = np.pad(
            data, 
            [(0, 0), (pad_len, pad_len), (0, 0)],
            constant_values=pad_value
        )
    
    logging.debug(data.shape)

    return data
    
def random_xy_flip(data, subj_i, seed=24, mode='slice'):
    logging.debug(data.shape)

    if mode == 'subject':
        data = tf.image.stateless_random_flip_up_down(data, seed=[subj_i, seed])
        data = tf.image.stateless_random_flip_left_right(data, seed=[subj_i, subj_i * seed])
    else:
        for slc_i in range(data.shape[-1]):
            slc = data[:,:,slc_i]
            slc = np.expand_dims(slc, axis=2)
            slc = tf.image.stateless_random_flip_up_down(slc, seed=[subj_i, slc_i * seed])
            slc = tf.image.stateless_random_flip_left_right(slc, seed=[subj_i, subj_i * slc_i * seed])
            slc = np.squeeze(slc)
            data[:,:,slc_i] = slc

    data = np.array(data)

    logging.debug(data.shape)

    return data

def standardize(data):
    logging.debug(data.shape)

    min_, max_ = np.min(data), np.max(data)
    num  = data - min_
    den  = max_ - min_
    data = num / den

    logging.debug(data.shape)

    return data

def threshold_intensities(data, value=5000):
    logging.debug(data.shape)
    orig_len = data.shape[-1]
    logging.debug(f'    Before: {orig_len}')

    if np.iscomplexobj(data):
        mag_data = np.abs(data)
        sum_intensities = np.array(
                [ np.sum(mag_data[:,:,i]) for i in range(mag_data.shape[-1]) ])
        data = data[:,:,np.where(sum_intensities > value)]
    else:
        sum_intensities = np.array([ np.sum(data[:,:,i]) for i in range(data.shape[-1]) ])
        data = data[:,:,np.where(sum_intensities > value)]

    new_len = data.shape[-1]
    logging.debug(f'    After: {new_len}')

    data = np.squeeze(data)

    return data

def white_noise(data_stack, mu=0.0, sigma=0.2):
    logging.debug(data_stack.shape)

    for subj_i in range(len(data_stack)):
        data = data_stack[subj_i]
        if type(sigma) == list:
            sigma = np.random.choice(sigma)

        if np.iscomplexobj(data):
            max_real, max_imag = np.max(np.real(data)), np.max(np.imag(data))
            real_noise_map = np.random.normal(mu, sigma * max_real, data.shape)
            imag_noise_map = 1j * np.random.normal(mu, sigma * max_imag, data.shape)
            data_stack[subj_i] = data + real_noise_map + imag_noise_map
        else:
            max_ = np.max(data)
            noise_map = np.random.normal(mu, sigma * max_, data.shape)
            data_stack[subj_i] = data + noise_map

    logging.debug(data_stack.shape)

    return data_stack

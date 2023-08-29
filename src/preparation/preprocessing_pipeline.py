import logging
import numpy as np
import tensorflow as tf
import pdb

def preprocess_data(data, params, steps, subj_i=None):
    '''
    Given a numpy array of a single subject, preprocess and return it.
    '''
    assert data.ndim == 3, f'Expected 3 dimensional data. Given {data.shape}'

    if np.iscomplexobj(data):
        data = data.astype('complex64')
    else:
        data = data.astype('float32')

    pipeline = gen_pipeline(steps=steps)
    pp_data = np.copy(data)
    for func, name in pipeline:
        if name == 'random_xy_flip':
            assert subj_i is not None
            pp_data = func(pp_data, subj_i, **params[name])
        elif params[name] is None:
            pp_data = func(pp_data)
        else:
            pp_data = func(pp_data, **params[name])

    return pp_data

def gen_pipeline(steps):
    """
    Return list of functions to apply onto data.
    """
    pipeline = []
    
    if steps is not None:
        for step in steps:
            if   step == 'normalize':
                pipeline.append( (normalize, step) )

            elif step == 'pad_square':
                pipeline.append( (pad_square, step) )

            elif step == 'random_xy_flip':
                pipeline.append( (random_xy_flip, step) )

            elif step == 'rescale_range':
                pipeline.append( (rescale_range, step) )

            elif step == 'standardize':
                pipeline.append( (standardize, step) )

            elif step == 'threshold_intensities':
                pipeline.append( (threshold_intensities, step) )

            elif step == 'white_noise':
                pipeline.append( (white_noise, step) )

            elif step in ['fourier_transform', 'ft']:
                pipeline.append( (fourier_transform, step) )

            elif step in ['inverse_fourier_transform', 'ift']:
                pipeline.append( (inverse_fourier_transform, step) )

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

    pad_len_0 = (384 - data.shape[0]) // 2
    pad_len_1 = (384 - data.shape[1]) // 2
    data = np.pad(
            data, 
            [(pad_len_0, pad_len_0), (pad_len_1, pad_len_1), (0, 0)],
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

def rescale_range(data, t_min, t_max):
    r_min, r_max = np.min(data), np.max(data)
    num = data - r_min
    den = r_max - r_min

    return (num / den) * (t_max - t_min) + t_min

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

def white_noise(data_slices, mu=0.0, sigma=0.2):
    logging.debug(data_slices.shape)

    for slc_i in range(data_slices.shape[-1]):
        data = data_slices[:,:,slc_i]
        if type(sigma) == list: sigma_ = np.random.choice(sigma)
        else: sigma_ = sigma

        if np.iscomplexobj(data):
            max_real, max_imag = np.max(np.real(data)), np.max(np.imag(data))
            real_noise_map = np.random.normal(mu, sigma_ * max_real, data.shape)
            imag_noise_map = 1j * np.random.normal(mu, sigma_ * max_imag, data.shape)
            data_slices[:,:,slc_i] = data + real_noise_map + imag_noise_map
        else:
            max_ = np.max(data)
            noise_map = np.random.normal(mu, sigma_ * max_, data.shape)
            data_slices[:,:,slc_i] = data + noise_map

    logging.debug(data_slices.shape)

    return data_slices

def fourier_transform(data, shift=True):
    '''
    Shift comes after the FT to operate on freq space.
    '''
    logging.debug(data.shape)

    data = np.fft.fft2(data, axes=(0, 1))

    if shift: data = np.fft.fftshift(data, axes=(0, 1))

    return data

def inverse_fourier_transform(data, shift=True):
    '''
    Shift comes before the FT to operate on freq space.
    '''
    logging.debug(data.shape)

    if shift: data = np.fft.ifftshift(data)

    data = np.fft.ifft2(data)

    return data

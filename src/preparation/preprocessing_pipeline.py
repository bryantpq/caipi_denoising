import logging
import numpy as np
import tensorflow as tf
import pdb

from patchify import patchify

from preparation.data_io import magphase2complex


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
        elif name not in params:
            pp_data = func(pp_data)
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
            if step == 'extract_patches':
                assert steps[-1] == step, 'Patchify must be last operation'
                pipeline.append( (extract_patches, step) )

            elif step in ['fourier_transform', 'ft']:
                pipeline.append( (fourier_transform, step) )

            elif step in ['inverse_fourier_transform', 'ift']:
                pipeline.append( (inverse_fourier_transform, step) )

            elif step == 'normalize':
                pipeline.append( (normalize, step) )

            elif step == 'pad_square':
                pipeline.append( (pad_square, step) )

            elif step == 'random_xy_flip':
                pipeline.append( (random_xy_flip, step) )

            elif step == 'rescale_magnitude':
                pipeline.append( (rescale_magnitude, step) )

            elif step == 'rescale_complex_frequency':
                pipeline.append( (rescale_complex_frequency, step) )

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

def extract_patches(data, dimensions=2, patch_size=[128,128,128], extract_step=[32,32,32]):
    assert dimensions == len(extract_step)
    assert dimensions == len(patch_size)

    if dimensions == 2:
        patch_size = patch_size + [1]
        extract_step = extract_step + [1]

    logging.debug(data.shape)
    
    patches = patchify(data, patch_size, step=extract_step)
    patches = patches.reshape(-1, *patch_size)

    if dimensions == 2: patches = np.squeeze(patches)

    logging.debug(patches.shape)

    return patches

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
    logging.debug('Pad square:')
    logging.debug(f'    Before: {data.shape}')

    largest_dim = max(data.shape[0], data.shape[1])

    pad_len_0 = (largest_dim - data.shape[0]) // 2
    pad_len_1 = (largest_dim - data.shape[1]) // 2
    data = np.pad(
            data, 
            [(pad_len_0, pad_len_0), (pad_len_1, pad_len_1), (0, 0)],
            constant_values=pad_value
        )
    
    logging.debug(f'    After: {data.shape}')

    return data
    
def random_xy_flip(data, subj_i, seed=24, mode='slice'):
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

    return data

def rescale_magnitude(data, t_min=0.0, t_max=1.0):
    r_min, r_max = np.min(data), np.max(data)
    num = data - r_min
    den = r_max - r_min

    return (num / den) * (t_max - t_min) + t_min

def rescale_complex_frequency(data, abs_min=-1.0, abs_max=1.0):
    '''
    Given some data in the frequency domain, compute the absolute value (magnitude) of the data
    and rescale this to a given range. Then recombine this with the angle value (phase) to create
    the rescaled complex frequency data.
    '''
    assert np.iscomplexobj(data)

    abs_data, ang_data = np.abs(data), np.angle(data)
    re_abs_data = rescale_magnitude(abs_data, abs_min, abs_max)

    return magphase2complex(re_abs_data, ang_data, rescale=False)

def threshold_intensities(data, value=5000):
    logging.debug(data.shape)
    orig_len = data.shape[-1]
    logging.debug(f'Slices in threshold_intensities:')
    logging.debug(f'    Before: {orig_len}')

    if np.iscomplexobj(data):
        mag_data = np.abs(data)
        sum_intensities = np.array(
                [ np.sum(mag_data[:,:,i]) for i in range(mag_data.shape[-1]) ]
        )
        data = data[:,:,np.where(sum_intensities > value)]
    else:
        sum_intensities = np.array(
                [ np.sum(data[:,:,i]) for i in range(data.shape[-1]) ]
        )
        data = data[:,:,np.where(sum_intensities > value)]

    logging.debug(list(sum_intensities))
    new_len = data.shape[-1]
    logging.debug(f'    After: {new_len}')

    data = np.squeeze(data)

    return data

def white_noise(data_slices, mu=0.0, sigma=0.2):
    '''
    Adds a different level of noise to each slice
    '''

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

    return data_slices

def low_pass_filter(img, window_size=64):
    assert window_size % 2 == 0
    assert np.iscomplexobj(img)
    assert img.ndim in [2, 3]
    
    mask = np.ones((window_size, ) * img.ndim, dtype=np.csingle)

    pad_len = [ ( (img.shape[i] - mask.shape[i]) // 2, ) * 2 for i in range(img.ndim) ]

    mask = np.pad(mask, pad_len, constant_values=0.0)
    
    return np.multiply(mask, img)

def fourier_transform(data, axes=(0, 1), shift=True):
    '''
    Apply 2D FT on first two dimensions of the given 3D volume.
    Shift comes after the FT to operate on freq space.
    '''
    assert np.iscomplexobj(data)
    logging.debug(data.shape)

    data = np.fft.fft2(data, axes=axes)
    if shift: data = np.fft.fftshift(data, axes=axes)

    return data

def inverse_fourier_transform(data, axes=(0, 1), shift=True):
    '''
    Shift comes before the FT to operate on freq space.
    '''
    assert np.iscomplexobj(data)
    logging.debug(data.shape)

    if shift: data = np.fft.ifftshift(data, axes=axes)
    data = np.fft.ifft2(data, axes=axes)

    return data

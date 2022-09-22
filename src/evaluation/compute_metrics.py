import logging
import math
import numpy as np
import pdb

from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as sk_mse
from math import log10, sqrt


def compute_metrics(X, y, vein_mask, wm_mask, lesion_mask, brain_mask=None, precision=None):
    '''
    Return dictionary of metrics between two images.
    '''
    X = X.astype(np.float64)
    y = y.astype(np.float64)

    metrics = {}

    metrics['mse']       = sk_mse(X.reshape(-1), y.reshape(-1))
    metrics['psnr']      = psnr(X, y)
    metrics['snr']       = snr(y)
    if brain_mask is not None: metrics['snr_brain'] = snr(y, brain_mask)
    metrics['ssim']      = ssim(X, y)
    metrics['luminance'] = luminance(X, y)
    metrics['contrast']  = contrast(X, y)
    metrics['structure'] = structure(X, y)
    metrics['cnr_vw']    = cnr(y, vein_mask, wm_mask)
    metrics['cnr_lv']    = cnr(y, vein_mask, lesion_mask)
    metrics['cnr_lw']    = cnr(y, wm_mask, lesion_mask)

    if precision is not None:
        for m in metrics.keys():
            metrics[m] = round(metrics[m], precision)

    return metrics

def snr(data, mask=None):
    '''
    Given an image, compute the SNR for the image.
    If a mask is provided, calculate SNR within the mask.
    '''
    if mask is not None:
        assert np.array_equal(np.unique(mask), [0, 1]) or \
                np.array_equal(np.unique(mask), [0]), 'Unexpected values for mask'
        data = data[np.where(mask == 1)]

    mu = np.mean(data)
    sd = np.std(data)

    return mu / sd

def psnr(X, y=None):
    '''
    If given two volumes, compute their pairwise psnr
    if given single volume, compute its psnr using background patches
    '''
    if y is not None: 
        return sk_psnr(X, y)

    patch_len = 25

    if X.ndim == 2:
        noise_vols = np.array( 
                [   X[:patch_len, :patch_len], X[-patch_len:, :patch_len],
                    X[:patch_len, -patch_len:], X[-patch_len:, -patch_len:]
                ]
        )
    elif X.ndim == 3:
        noise_vols = np.array(
                [   X[:patch_len,:patch_len,:patch_len], 
                    X[:patch_len,:patch_len,-patch_len:],
                    X[:patch_len,-patch_len:,:patch_len],
                    X[-patch_len:,:patch_len,:patch_len],
                    X[:patch_len,-patch_len:,-patch_len:],
                    X[-patch_len:,:patch_len,-patch_len:],
                    X[-patch_len:,-patch_len:,:patch_len],
                    X[-patch_len:,-patch_len:,-patch_len:]
                ]
        )

    noise_std = np.std(noise_vols.reshape(-1))
    max_intensity = 1.0

    result = 10 * log10(max_intensity / noise_std)

    return result

def cnr(data, mask1, mask2):
    '''
    Given (data, mask1, mask2), extract voxels from data at mask1 & mask2, and compute
    CNR betwen the two volumes.
    '''
    assert np.array_equal(np.unique(mask1), [0, 1]) or \
            np.array_equal(np.unique(mask1), [0]), 'Unexpected values for mask1'
    assert np.array_equal(np.unique(mask2), [0, 1]) or \
            np.array_equal(np.unique(mask2), [0]), 'Unexpected values for mask2'

    masked_values1 = data[np.where(mask1 == 1)]
    masked_values2 = data[np.where(mask2 == 1)]

    num = abs(np.sum(masked_values1) - np.sum(masked_values2))
    den = len(masked_values1) + len(masked_values2)

    if den == 0: return None

    return num / den

def luminance(X, y):
    c1 = math.pow(0.01 * 1.0, 2)
    mean_X, mean_y = np.mean(X), np.mean(y)
    num = 2 * mean_X * mean_y + c1
    den = math.pow(mean_X, 2) + math.pow(mean_y, 2) + c1

    if den == 0: return None

    return num / den

def contrast(X, y):
    c2 = math.pow(0.03 * 1.0, 2)
    std_X, std_y = np.std(X), np.std(y)

    num = 2 * std_X * std_y + c2
    den = math.pow(std_X, 2) + math.pow(std_y, 2) + c2

    if den == 0: return None

    return num / den

def structure(X, y):
    c3 = math.pow(0.03 * 1.0, 2) / 2
    cov_mat = np.cov(X.reshape(-1), y.reshape(-1))
    cov_Xy = cov_mat[0,1]
    std_X, std_y = np.std(X), np.std(y)
    num = cov_Xy + c3
    den = std_X * std_y + c3

    if den == 0: return None

    return num / den

def my_mse(X, y):
    total = np.sum(np.power(X - y, 2))
    N = np.product(X.shape)

    return total / N

def my_psnr(X, y):
    MAX_INTENSITY = 1.0
    m = my_mse(X, y)

    return 10 * math.log10(MAX_INTENSITY / m)

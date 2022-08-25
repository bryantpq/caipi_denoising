import logging
import math
import numpy as np
import pdb

from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse
from math import log10, sqrt


def compute_metrics(X, y):
    '''
    Return dictionary of metrics between two images.
    Metrics
        MSE
        PSNR
        SSIM
    '''
    metrics = {}

    if X.ndim == 2:
        metrics['mse'] = mse(X, y)

    metrics['psnr'] = round(psnr(X, y), 3)
    metrics['ssim'] = round(ssim(X, y), 3)

    return metrics

def psnr(X, y=None):
    '''
    If given two volumes, compute their pairwise psnr
    if given single volume, compute its psnr using background patches
    '''
    X = X.astype(np.float64)

    if y is not None: 
        y = y.astype(np.float64)
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

        #noise_img = np.resize(noise_vols.reshape(-1), 384*312*256).reshape((384,312,256))
        #res = sk_psnr(X, noise_img)

    noise_std = np.std(noise_vols.reshape(-1))
    print(noise_std)
    max_intensity = 1.0

    # print(f'noise std: {noise_std}')
    result = 10 * log10(max_intensity / noise_std)

    return result

def cnr(data, mask1, mask2):
    '''
    Given (data, mask1, mask2), extract voxels from data at mask1 & mask2, and compute
    CNR betwen the two volumes.
    '''
    assert np.array_equal(np.unique(mask1), [0, 1]), 'Unexpected values for mask1'
    assert np.array_equal(np.unique(mask2), [0, 1]), 'Unexpected values for mask2'

    data = data.astype(np.float64)
    masked_values1 = data[np.where(mask1 == 1)]
    masked_values2 = data[np.where(mask2 == 1)]

    num = abs(np.sum(masked_values1) - np.sum(masked_values2))
    den = len(masked_values1) + len(masked_values2)

    return num / den

def luminance(X, y):
    c1 = math.pow(0.01 * 1.0, 2)
    mean_X, mean_y = np.mean(X), np.mean(y)
    num = 2 * mean_X * mean_y + c1
    den = math.pow(mean_X, 2) + math.pow(mean_y, 2) + c1

    return num / den

def contrast(X, y):
    c2 = math.pow(0.03 * 1.0, 2)
    std_X, std_y = np.std(X), np.std(y)
    num = 2 * std_X * std_y + c2
    den = math.pow(std_X, 2) + math.pow(std_y, 2) + c2

    return num / den

def structure(X, y):
    c3 = math.pow(0.03 * 1.0, 2) / 2
    cov_Xy = np.cov(X.reshape(-1), y.reshape(-1))[0,0]
    std_X, std_y = np.std(X), np.std(y)
    num = cov_Xy + c3
    den = std_X * std_y + c3

    return num / den

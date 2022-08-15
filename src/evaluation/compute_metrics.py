import logging
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse


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

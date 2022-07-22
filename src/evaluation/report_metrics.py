import logging
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def report_metrics(X, y):
    all_psnr = []
    all_ssim = []

    for i in range(len(X)):
        X_slc, y_slc = X[i], y[i]

        all_psnr.append(psnr(X_slc, y_slc))
        all_ssim.append(ssim(X_slc, y_slc))

    logging.info('PSNR values:')
    logging.info(all_psnr)
    logging.info('SSIM values:')
    logging.info(all_ssim)

import logging
import math
import numpy as np
import pdb

from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as sk_mse
from math import log10, sqrt

from preparation.preprocessing_pipeline import rescale_magnitude

# SNR functions
# snr - unary whole volume SNR
# axial_psnr - unary axial snr using white matter mask
# psnr - binary snr using sk_learn
# my_psnr - binary psnr using mse formula

SUBJ_RESTRICT_RANGE = {
        '1_01_024-V1': (75, 384),
        '1_01_029-V1': (0, 300),
        '1_01_040-V1': (100, 384),
        '1_07_003-V1': (50, 384),
        '1_07_005-V1': (50, 260),
        '1_07_009-V1': (50, 200),
        '1_07_010-V1': (0, 200),
        '1_07_011-V1': (50, 384),
        '1_07_017-V1': (60, 384),
        '1_07_022-V1': (60, 384),
        '1_07_025-V1': (70, 384),
        '1_07_028-V1': (70, 384),
        '1_07_038-V1': (80, 384),
        '1_07_039-V1': (60, 384),
        '1_08_046-V1': (80, 384),
        '1_08_047-V1': (0, 250),
        'CID145': (80, 384),
        'CID149': (50, 234),
        'CID191': (0, 234)
        }

def compute_metrics(reference, before, after, vein_mask, wm_mask, lesion_mask, 
        brain_mask=None, 
        precision=None,
        subj_id=None):
    '''
    Return dictionary of metrics between two images.
    '''
    reference = reference.astype(np.float64)
    before = before.astype(np.float64)
    after = after.astype(np.float64)

    before_metrics = {}
    after_metrics  = {}

    before_metrics['axial_psnr'] = axial_psnr(before, mask=wm_mask, debug=True, subj_id=subj_id)
    before_metrics['ssim']       = 'N/A' #ssim(reference, before)
    before_metrics['luminance']  = 'N/A' #luminance(reference, before)
    before_metrics['contrast']   = 'N/A' #contrast(reference, before)
    before_metrics['structure']  = 'N/A' #structure(reference, before)

    after_metrics['axial_psnr'] = axial_psnr(after, mask=wm_mask, debug=True, subj_id=subj_id)
    after_metrics['ssim']       = ssim(before, after)
    after_metrics['luminance']  = luminance(before, after)
    after_metrics['contrast']   = contrast(before, after)
    after_metrics['structure']  = structure(before, after)

    if precision is not None:
        for m in before_metrics.keys():
            before_metrics[m] = round(before_metrics[m], precision)
        for m in after_metrics.keys():
            after_metrics[m] = round(after_metrics[m], precision)

    return before_metrics, after_metrics

    #before_metrics['mse']        = sk_mse(reference.reshape(-1), before.reshape(-1))
    #before_metrics['snr']        = snr(before)
    #before_metrics['cnr_vw']    = cnr(before, vein_mask, wm_mask)
    #before_metrics['cnr_lv']    = cnr(before, vein_mask, lesion_mask)
    #before_metrics['cnr_lw']    = cnr(before, wm_mask, lesion_mask)

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
    sd = np.std(data)**2

    return mu / sd

def volume_psnr(y, mask=None, debug=False, subj_id=None):
    if mask is not None:
        cylinder_mask = _create_axial_cylinder_mask()
        cylinder_mask[np.where(mask == 0)] = 0
        assert np.array_equal(np.unique(mask), [0, 1]) or \
               np.array_equal(np.unique(mask), [0]), 'Unexpected values for mask'
    
    wm_brain = np.multiply(y, cylinder_mask)
    wm_brain = wm_brain[np.where(cylinder_mask == 1)]

    max_int = np.max(wm_brain)
    std_int = np.std(wm_brain)

    res = max_int / std_int

    if isinstance(res, complex):
        res = np.abs(res)

    if debug:
        return res, wm_brain, max_int, std_int
    else:
        return res

def axial_psnr(y, mask=None, debug=False, subj_id=None):
    '''
    Given a data volume and mask for a region of interest (white matter),
    create a cylinder through the axial plane and calculate the snr for regions
    intersecting within these 3 volumes. SNR for 2D slices is calculated along the axial plane.
    Among these 2D SNR calculations, return the highest value.
    '''
    if mask is not None:
        cylinder_mask = _create_axial_cylinder_mask()
        cylinder_mask[np.where(mask == 0)] = 0
        assert np.array_equal(np.unique(mask), [0, 1]) or \
               np.array_equal(np.unique(mask), [0]), 'Unexpected values for mask'
    
    # TODO
    # fill the axial slices with indices we dont want with 0's
    if subj_id in SUBJ_RESTRICT_RANGE:
        START_I, END_I = SUBJ_RESTRICT_RANGE[subj_id]
    else:
        START_I, END_I = 0, y.shape[0]

    pre_masked_snrs  = np.array([ -1 for i in range(0, START_I) ])
    post_masked_snrs = np.array([ -1 for i in range(END_I, 384) ])
    slice_snr = np.array([ snr(y[ii], cylinder_mask[ii]) for ii in range(START_I, END_I) ])

    slice_snr = np.append(pre_masked_snrs, slice_snr)
    slice_snr = np.append(slice_snr, post_masked_snrs)

    slice_snr[np.isnan(slice_snr)] = 0
    slice_snr[np.isinf(slice_snr)] = 0

    max_snr_idx = np.argmax(slice_snr)
    max_snr = slice_snr[max_snr_idx]

    if isinstance(max_snr, complex):
        max_snr = np.log(np.abs(max_snr)) * 10

    max_snr = 10 * math.log10(max_snr)

    if debug:
        return max_snr, slice_snr, cylinder_mask
    else:
        return max_snr

def _create_axial_cylinder_mask(rad=0.5, size=(384, 312, 256)):
    xx = np.linspace(-1, 1, size[2])
    yy = np.linspace(-1, 1, size[1])

    xs, ys = np.meshgrid(xx, yy)
    zz = np.sqrt(xs**2 + ys**2)

    mask2d = np.zeros(size[1:])
    mask2d[np.where(zz < rad)] = 1

    mask3d = np.zeros(size)
    for i in range(size[0]): mask3d[i] = mask2d

    mask3d = np.swapaxes(mask3d, 0, 1)

    return mask3d

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
    MAX_INTENSITY = np.max(X)
    m = my_mse(X, y)

    return 10 * math.log10(MAX_INTENSITY / m)

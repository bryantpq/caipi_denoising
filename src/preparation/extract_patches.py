import logging
import utils.istarmap
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import tqdm

from utils.data_io import write_patches
from sklearn.feature_extraction.image import extract_patches_2d


def extract_patches(data, 
                    X_OR_Y,
                    save_path,
                    patch_size=(32, 32), 
                    extract_step=(1, 1), 
                    pad_before_ext=True, 
                    pad_value=0.0,
                    return_patches=False,
                    save_dtype='float16',
                    workers=32):
    
    size_x, size_y = patch_size
    step_x, step_y = extract_step

    if pad_before_ext:
        data = np.pad(
                data, 
                [(0, 0), (size_x // 2, size_x // 2), (size_y // 2, size_y // 2), (0, 0)], 
                constant_values=pad_value
        )
    
    n_patches_per_slice = len(extract_patches_2d(data[0], patch_size))

    indices = []
    for y_i in range(0, n_patches_per_slice, (data.shape[2] - size_y + 1) * step_y):
        for x_i in range(0, data.shape[2] - size_x + 1, step_x):
            indices.append(y_i + x_i)
  
    logging.info('Estimating space required to save patches...')
    logging.info('Assuming data is float16 = 2bytes per pixel')
    logging.info('    Number of slices:    {}'.format(len(data)))
    logging.info('    Patches per slice:   {}'.format(len(indices)))
    logging.info('    Patch size:          {}'.format(patch_size))
    logging.info('    Save data type:      {}'.format(save_dtype))
    logging.info('        Per slice: {} * {} * 2 = {:.2e} bytes'.format(len(indices), patch_size, 
                                                       len(indices) * patch_size[0] * patch_size[1] * 2))
    logging.info('            Total: {} * {} * {} * 2 = {:.2e} bytes'.format(len(data), len(indices), patch_size, 
                                                        len(data) * len(indices) * patch_size[0] * patch_size[1] * 2))
    logging.info('')
    
    res = []
    with mp.Pool(workers) as pool:
        args_list = []
        for ii, slc in enumerate(data):
            args = [
                slc[:,:,0],
                ii,
                indices,
                patch_size,
                X_OR_Y,
                save_dtype,
                save_path,
                return_patches
            ]
            args_list.append(args)

        logging.info('Saving patches to {} ...'.format(save_path))
        res = list(tqdm.tqdm(pool.istarmap(_extract_patches_from_slice, args_list), total=len(args_list)), ncols=80)

    logging.info(f'Completed. Generated {len(res)} patches for {X_OR_Y}.')

    if return_patches:
        return np.vstack(res)
    else:
        return res

def _extract_patches_from_slice(slc,
                                slc_i,
                                keep_idx,
                                patch_size,
                                X_OR_Y,
                                save_dtype,
                                save_path,
                                return_patches):
    patches = extract_patches_2d(slc, patch_size)
    patches = np.expand_dims(patches, axis=3)
    patches = patches[keep_idx]
    
    write_patches(slc_i,
                  patches, 
                  X_OR_Y,
                  save_dtype, 
                  save_path)

    if return_patches:
        return patches
    else:
        return slc_i

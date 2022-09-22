import logging
import utils.istarmap
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import tqdm

from patchify import patchify
from utils.data_io import write_patches


def extract_patches(data, 
                    X_OR_Y,
                    save_path,
                    dimensions=3,
                    patch_size=(32, 32), 
                    extract_step=(1, 1), 
                    pad_before_ext=False, 
                    pad_value=0.0,
                    save_dtype='float16',
                    workers=32,
                    return_patches=False):
    
    logging.debug(data.shape)
    if pad_before_ext:
        if dimensions == 2:
            size_x, size_y = patch_size
            data = np.pad(
                    data, 
                    [(0, 0), (size_x // 2, size_x // 2), (size_y // 2, size_y // 2), (0, 0)], 
                    constant_values=pad_value
            )
        elif dimensions == 3:
            size_x, size_y, size_z = patch_size
            data = np.pad(
                    data, 
                    [(0, 0), (size_x // 2, size_x // 2), (size_y // 2, size_y // 2), (size_z // 2, size_z // 2)], 
                    constant_values=pad_value
            )
        logging.debug(data.shape)
    
    logging.debug(data[0,:,:,0].shape)
    if dimensions == 2:
        slc0_patches = patchify(data[0,:,:,0], patch_size, step=extract_step)
    elif dimensions == 3:
        vol0_patches = patchify(data[:256,:,:,0], patch_size, step=extract_step)
    logging.debug(slc0_patches.shape)
    n_patches_per_slice = slc0_patches.shape[0] * slc0_patches.shape[1]

    logging.debug('Extracting patches from dataset: {}'.format(data.shape))
    logging.debug('Estimating space required to save patches...')
    logging.debug('Assuming data is float16 = 2bytes per pixel')
    logging.debug('  Number of slices:    {}'.format(len(data)))
    logging.debug('  Patches per slice:   {}'.format(n_patches_per_slice))
    logging.debug('  Patch size:          {}'.format(patch_size))
    logging.debug('  Save data type:      {}'.format(save_dtype))
    logging.debug('    Per slice: {} * {} * 2 = {:.2e} bytes'.format(n_patches_per_slice, patch_size, 
            n_patches_per_slice * patch_size[0] * patch_size[1] * 2))
    logging.info('   Total: {} * {} * {} * 2 = {:.2e} bytes'.format(len(data), n_patches_per_slice, patch_size, 
            len(data) * n_patches_per_slice * patch_size[0] * patch_size[1] * 2))
    logging.debug('')
    
    res = []
    with mp.Pool(workers) as pool:
        args_list = []
        for ii, slc in enumerate(data):
            args = [
                slc[:,:,0],
                ii,
                patch_size,
                extract_step,
                X_OR_Y,
                save_dtype,
                save_path,
                return_patches
            ]
            args_list.append(args)

        logging.info('Saving patches to {} ...'.format(save_path))
        res = list( tqdm.tqdm(
            pool.istarmap(_extract_patches_from_slice, args_list),
            total=len(args_list), ncols=80) 
        )

    logging.info(f'Completed. Generated {len(data) * n_patches_per_slice} patches for {X_OR_Y}.')

    if return_patches:
        return np.vstack(res)
    else:
        return res

def _extract_patches_from_slice(slc,
                                slc_i,
                                patch_size,
                                extract_step,
                                X_OR_Y,
                                save_dtype,
                                save_path,
                                return_patches):
    patches = patchify(slc, patch_size, step=extract_step)
    patches = patches.reshape(-1, *patch_size)
    patches = np.expand_dims(patches, axis=3)
    
    write_patches(slc_i,
                  patches, 
                  X_OR_Y,
                  save_dtype, 
                  save_path)

    if return_patches:
        return patches
    else:
        return slc_i

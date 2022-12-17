import logging
import utils.istarmap
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import tqdm
import pdb

from patchify import patchify
from utils.data_io import write_data

def extract_patches(
        data,
        filename_placeholder,
        patch_size=(32, 32), 
        extract_step=(1, 1), 
        pad_before_ext=False, 
        pad_value=0.0,
        save_dtype='float16',
        workers=32
    ):

    # collect all slices into an np array
    # collapse first two dimensions to get array of slices
    data = np.vstack([ np.moveaxis(d, -1, 0) for d in data ])
    logging.info('  data.shape: {}'.format(data.shape))

    if pad_before_ext:
        size_x, size_y = patch_size
        for subj_data in data:
            data = np.pad(
                    data,
                    [(0, 0), (size_x // 2, size_x // 2), (size_y // 2, size_y // 2), (0, 0)], 
                    constant_values=pad_value
            )
            logging.info(data.shape)

    first_slc_patches = patchify(data[0,:,:], patch_size, step=extract_step)
    n_patches_per_slice = first_slc_patches.shape[0] * first_slc_patches.shape[1]
    logging.debug('Extracting patches from dataset: {}'.format(data.shape))
    logging.debug('Estimating space required to save patches...')
    logging.debug('Assuming data is float16 = 2bytes per pixel')
    logging.debug('  Number of slices:    {}'.format(len(data)))
    logging.debug('  Patches per slice:   {}'.format(n_patches_per_slice))
    logging.debug('  Patch size:          {}'.format(patch_size))
    logging.info('  Save data type:      {}'.format(save_dtype))
    logging.debug('    Per slice: {} * {} * 2 = {:.2e} bytes'.format(
            n_patches_per_slice, patch_size, 
            n_patches_per_slice * patch_size[0] * patch_size[1] * 2))
    logging.info('   Space required: {} * {} * {} * 2 = {:.2e} bytes'.format(
            len(data), n_patches_per_slice, patch_size, 
            len(data) * n_patches_per_slice * patch_size[0] * patch_size[1] * 2))
    logging.debug('')
    
    res = []
    with mp.Pool(workers) as pool:
        args_list = []
        for slc_i, slc_data in enumerate(data):
            filename = filename_placeholder.format(slc_i)
            args = [
                    slc_data,
                    filename,
                    patch_size,
                    extract_step,
                    save_dtype,
            ]
            args_list.append(args)

        folder_path = '/'.join(filename_placeholder.split('/')[:-1])
        logging.info(f'    Saving patches to {folder_path}...')
        res = list(tqdm.tqdm(
            pool.istarmap(_extract_patches, args_list),
            total=len(args_list), ncols=80)
        )
    logging.info(f'    Completed. Generated {len(data) * n_patches_per_slice} patches.')

def _extract_patches(
        slc_data,
        filename,
        patch_size,
        extract_step,
        save_dtype
    ):
    patches = patchify(slc_data, patch_size, step=extract_step)
    patches = patches.reshape(-1, *patch_size)

    write_data(patches, filename, save_dtype)

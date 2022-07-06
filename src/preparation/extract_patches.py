import multiprocessing as mp
import numpy as np
import tensorflow as tf

from utils.data_io import write_patches
from sklearn.feature_extraction.image import extract_patches_2d


def extract_patches(data, 
                    X_OR_Y,
                    save_path,
                    patch_size=(32, 32), 
                    extract_step=(1, 1), 
                    pad_before_ext=True, 
                    pad_value=0.0,
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
  
    print()
    print('Estimating space required to save patches...')
    print('Assuming data is float16 = 2bytes per pixel')
    print('    Number of slices:    ', len(data))
    print('    Patches per slice:   ', len(indices))
    print('    Patch size:          ', patch_size)
    print('    Save data type:      ', save_dtype)
    print('        Total: {} * {} * {} * 2 = {:e} bytes'.format(len(data), len(indices), patch_size, 
                                                        len(data) * len(indices) * patch_size[0] * patch_size[1] * 2))
    print('        Per slice: {} * {} * 2 = {:e} bytes'.format(len(indices), patch_size, 
                                                       len(indices) * patch_size[0] * patch_size[1] * 2))
    print()
    
    pool = mp.Pool(workers)
    processes = []
    for ii, slc in enumerate(data):
        slc_ = slc[:,:,0]
        processes.append( pool.apply_async(_extract_patches_from_slice,
                                           args=(slc_, ii, indices, patch_size, X_OR_Y, save_dtype, save_path)) )
        
    print('Saving patches to {}'.format(save_path))
    results = [p.get() for p in processes] 

    return np.vstack(results)


def _extract_patches_from_slice(slc,
                                slc_i,
                                keep_idx,
                                patch_size,
                                X_OR_Y,
                                save_dtype,
                                save_path):
    
    patches = extract_patches_2d(slc, patch_size)
    patches = np.expand_dims(patches, axis=3)
    patches = patches[keep_idx]
    
    if slc_i % 500 == 0: 
        print('    ... completed slice {}'.format(slc_i))

    write_patches(slc_i,
                  patches, 
                  X_OR_Y,
                  save_dtype, 
                  save_path)
        
    return patches

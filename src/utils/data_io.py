import logging
import multiprocessing as mp
import numpy as np
import os

"""
Save/Load by slices/patches rather than whole array to track progress
"""

def load_dataset(data_folder):
    files = os.listdir(data_folder)
    
    if len(files) == 2:
        logging.info('Loading slices...')
        X_file = os.path.join(data_folder, files[0])
        y_file = os.path.join(data_folder, files[1])
        
        return np.load(X_file), np.load(y_file)
    
    elif len(files) > 2: 
        logging.info('Loading patches...')
        X_patches = load_patches('X', data_folder)
        y_patches = load_patches('y', data_folder)
        
        return X_patches, y_patches
    
    else:
        logging.info('Error not understood number of files in dir: {}'.format(data_folder))
        
        return None, None
        

def write_slices(slices,
                 X_OR_Y,
                 save_path, 
                 save_dtype):
    create_folders(save_path)
    save_path = os.path.join(save_path, X_OR_Y + '.npy')
    slices = slices.astype(save_dtype)
    
    np.save(save_path, slices)

    
def write_patches(slc_i, 
                  patches,
                  X_OR_Y,
                  save_dtype='float16',
                  save_path='/home/quahb/caipi_denoising/data/preprocessed_patches/config1'):
    create_folders(save_path)
    save_path = os.path.join(save_path, str(slc_i) + '_{}.npy'.format(X_OR_Y))
    patches = patches.astype(save_dtype)

    np.save(save_path, patches)  


def load_patches(X_OR_Y, folder_path, workers=32):
    
    files = [ f for f in os.listdir(folder_path) if X_OR_Y in f.split('.')[0] ]
    logging.info('    Found {}{} files to load at {}'.format(len(files), X_OR_Y, folder_path))

    # sort file names
    files = [ ( int(fname.split('_')[0]), fname ) for fname in files ]
    files.sort(key=lambda x: x[0])
    files = [ f[1] for f in files ]

    pool = mp.Pool(workers, maxtasksperchild=1)
    processes = []
    
    for f in files:
        f_path = os.path.join(folder_path, f)
        processes.append( pool.apply_async(np.load,
                                           args=(f_path, )) )
    results = [ p.get() for p in processes ]
    
    results = np.vstack(results)
    
    logging.info('    Loading patches complete.')
    logging.info('    Dataset shape: {}'.format(results.shape))
    
    return results


def create_folders(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logging.info("Creating new folder dataset: {}".format(path))

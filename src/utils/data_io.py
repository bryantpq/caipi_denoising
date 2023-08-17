import logging
import multiprocessing as mp
import numpy as np
import os
import pdb
from tqdm import tqdm

"""
Save/Load by slices/patches rather than whole array to track progress
"""
def load_dataset(data_folder, load_folds=None, postprocess_mode=None):
    files = os.listdir(data_folder)

    if load_folds not in [None, False, -1]:
        # filter folds not to be loaded
        if type(load_folds) != list: load_folds = [ load_folds ]
        keep_files = []
        for fname in files:
            fold_i_with_extension = fname.split('_')[-1] # fXX.npy
            fold_i = int(fold_i_with_extension.split('.')[0][1:])
            if fold_i in load_folds: keep_files.append(fname)
        files = keep_files

    files = sorted(files)

    data = []
    for f in tqdm(files, ncols=100):
        data.append( np.load(os.path.join(data_folder, f)) )

    if postprocess_mode == 'train':
        if len(files[0].split('_')) == 2: # process patches
            data = np.vstack(data)
        else: # process volumes
            data = [ np.moveaxis(d, -1, 0) for d in data ]
            data = np.vstack(data)

        data = np.expand_dims(data, axis=3) # becomes [n, X, Y, 1]
        return data
    elif postprocess_mode == 'test':
        data = np.stack(data)

        return data, files


def write_data(data, filename, save_dtype):
    create_folders('/'.join(filename.split('/')[:-1]))
    data = data.astype(save_dtype)

    np.save(filename, data)
    

def create_folders(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

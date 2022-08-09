import json
import logging
import os
from tqdm import tqdm

import numpy as np
import pydicom as dicom


def get_train_data(train_loo=False):
    """
    Return np.array of all slices to be used for training and validation.
    """
    MODALITY = '3D_T2STAR_segEPI'
    dicoms_dict = get_data_dict()
    
    X_train, slc_paths = get_slices(MODALITY, dicoms_dict)
    X_train = np.expand_dims(X_train, axis=3)
    y_train = np.copy(X_train)
    
    if train_loo is not False:
        train, test = train_leave_one_out(X_train, y_train, slc_paths)

        if train_loo == 'train':
            return train
        elif train_loo == 'test':
            return test

    return X_train, y_train, slc_paths


def train_leave_one_out(X, y, paths, seed=24):
    """
    Split given lists by number of subjects into 80:20 ratio
    """
    split_i = int(63 * 0.8)
    shuffle_i = np.random.RandomState(seed=seed).permutation(int(len(X) / 256))
    train_subj_i, test_subj_i = shuffle_i[:split_i], shuffle_i[split_i:]

    train_i = []
    for subj_i in train_subj_i:
        train_i.extend(list(range(subj_i*256, subj_i*256 + 256)))

    test_i = []
    for subj_i in test_subj_i:
        test_i.extend(list(range(subj_i*256, subj_i*256 + 256)))

    train_paths = [paths[i] for i in train_i]
    test_paths  = [paths[i] for i in test_i]

    id_pos = 6
    train_subj = []
    for i in range(0, len(train_i), 256):
        subj_id = train_paths[i].split('/')[id_pos]
        train_subj.append(subj_id)

    test_subj = []
    for i in range(0, len(test_i), 256):
        subj_id = test_paths[i].split('/')[id_pos]
        test_subj.append(subj_id)

    logging.info(f'Training indices: {train_subj_i}')
    logging.info(f'Training subjects: {train_subj}')
    logging.info('')
    logging.info(f'Testing indices: {test_subj_i}')
    logging.info(f'Testing subjects: {test_subj}')

    train = [X[train_i], y[train_i], train_paths]
    test  = [ X[test_i],  y[test_i],  test_paths]

    return train, test


def get_test_data():
    """
    Return np.array of all slices to be used for testing.
    """
    MODALITIES = ['CAIPI1x2', 'CAIPI1x3', 'CAIPI2x2']
    dicoms_dict = get_data_dict()
    
    to_stack  = []
    slc_paths = []
    for m in MODALITIES:
        arr, paths = get_slices(m, dicoms_dict)
        to_stack.append(arr)
        slc_paths.extend(paths)
    
    X_test = np.vstack(to_stack)
    X_test = np.expand_dims(X_test, axis=3)
    
    return X_test, slc_paths


def get_data_dict(json_file_path='/home/quahb/caipi_denoising/data/data.json'):
    """
    Create dict for 
        {subj_id:
            {modality: 
                [dicom_paths]
            }
        }
    """
    dicoms_dict = {}
    
    with open(json_file_path) as json_file:
        data_json = json.load(json_file)

        for subj in data_json['subjects']:
            dicoms_dict[subj] = {}
            for modal in data_json['modalities']:
                path = '/home/quahb/caipi_denoising/data/source_dicoms/{}/{}/'.format(subj, modal)
                subj_dicoms = os.listdir(path)
                for i in range(len(subj_dicoms)):
                    subj_dicoms[i] = path + subj_dicoms[i]
                dicoms_dict[subj][modal] = subj_dicoms

    return dicoms_dict


def get_slices(modality, dicoms_dict):
    """
    Return np.array of all dicom slices for a single modality. Dicom slices are sorted subject-wise.
    """
    all_slices = []
    all_paths  = []
    for subj in tqdm(dicoms_dict.keys(), ncols=80):
        subj_slices_path = dicoms_dict[subj][modality]
        subj_slices = []
        for slc_path in subj_slices_path:
            ds = dicom.dcmread(slc_path)
            subj_slices.append(ds)

        subj_slices = [ slices for slices, _ in sorted( 
            zip(subj_slices, subj_slices_path), 
            key=lambda pair: pair[0].SliceLocation ) 
        ]
        subj_slices_path = [ paths for _, paths in sorted( 
            zip(subj_slices, subj_slices_path), 
            key=lambda pair: pair[0].SliceLocation ) 
        ]

        subj_slices = [s.pixel_array for s in subj_slices]

        all_slices.extend([s for s in subj_slices])
        all_paths.extend(subj_slices_path)
    
    return np.stack(all_slices, axis=0), all_paths


def get_median_slices(X, 
                      left_i=35, 
                      right_i=221):
    """
    Given a np.array of the sorted slices, return a np.array for only the median slices
    """
    N_SLCS = 256
    n_subjects = X.shape[0] // N_SLCS

    outputs = np.empty((0, X.shape[1], X.shape[2], X.shape[3]))
    for i in range(n_subjects):
        outputs = np.vstack(
            [ outputs, X[i * N_SLCS + left_i: i * N_SLCS + right_i] ]
        )

    return outputs

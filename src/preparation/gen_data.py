import json
import os

import numpy as np
import pydicom as dicom


def get_train_data(median_slices=False):
    """
    Return np.array of all slices to be used for training and validation.
    """
    MODALITY = '3D_T2STAR_segEPI'
    dicoms_dict = get_data_dict()
    
    X_train, _ = get_slices(MODALITY, dicoms_dict)
    X_train = np.expand_dims(X_train, axis=3)
    y_train = np.copy(X_train)
    
    if median_slices:
        X_train = get_median_slices(X_train)
        y_train = get_median_slices(y_train)
    
    
    return X_train, y_train


def get_test_data():
    """
    Return np.array of all slices to be used for testing.
    """
    MODALITIES = ['CAIPI1x2', 'CAIPI1x3', 'CAIPI2x2']
    dicoms_dict = get_data_dict()
    
    to_stack = []
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
    for subj in dicoms_dict.keys():
        subj_slices_path = dicoms_dict[subj][modality]
        subj_slices = []
        for slc_path in subj_slices_path:
            ds = dicom.dcmread(slc_path)
            subj_slices.append(ds)
        
        subj_slices = [ slices for slices, _ in sorted( zip(subj_slices, subj_slices_path), key=lambda pair: pair[0].SliceLocation ) ]
        subj_slices_path = [ paths for _, paths in sorted( zip(subj_slices, subj_slices_path), key=lambda pair: pair[0].SliceLocation ) ]
        all_slices.extend([s.pixel_array for s in subj_slices])
        all_paths.extend(subj_slices_path)
    
    return np.stack(all_slices, axis=0), all_paths


def get_median_slices(X, 
                      n_subjects=None, 
                      left_i=35, 
                      right_i=221):
    """
    Given a np.array of the sorted slices, return a np.array for only the median slices
    """
    N_SLCS = 256
    
    if n_subjects is None:
        n_subjects = X.shape[0] // N_SLCS
    else:
        n_subjects = 63
        
    outputs = np.empty((0, X.shape[1], X.shape[2], X.shape[3]))
    for i in range(n_subjects):
        outputs = np.vstack(
            [ outputs, X[i * N_SLCS + left_i: i * N_SLCS + right_i] ]
        )

    return outputs
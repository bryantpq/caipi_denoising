import json
import logging
import nibabel as nib
import os
import pdb
from tqdm import tqdm
from sklearn.model_selection import KFold

import numpy as np
import pydicom as dicom

DATA_FILE = '/home/quahb/caipi_denoising/data/data_v2.json'
N_SUBJECTS = 52

def get_train_data(dimensions, train_loo=False):
    """
    Return np.array of all slices to be used for training and validation.
    """
    MODALITY = '3D_T2STAR_segEPI'
    data_dict = get_data_dict()
    
    X_train, slc_paths = _get_dicoms(dimensions, MODALITY, data_dict)

    _test_data_shape(dimensions, X_train)

    y_train = np.copy(X_train)
    
    if train_loo is not False:
        if dimensions == 2:
            train, test = train_leave_one_out_2D(X_train, y_train, slc_paths)
        elif dimensions == 3:
            train, test = train_leave_one_out_3D(X_train, y_train, slc_paths)

        if train_loo == 'train':
            return train
        elif train_loo == 'test':
            return test

    return X_train, y_train, slc_paths


def get_test_data(dimensions):
    """
    Return np.array of all slices to be used for testing.
    """
    MODALITIES = ['CAIPI1x2', 'CAIPI1x3', 'CAIPI2x2']
    data_dict = get_data_dict()
    
    to_stack  = []
    slc_paths = []
    for m in MODALITIES:
        arr, paths = _get_dicoms(dimensions, m, data_dict)
        to_stack.append(arr)
        slc_paths.extend(paths)
    
    X_test = np.vstack(to_stack)

    _test_data_shape(dimensions, X_test)

    return X_test, slc_paths


def get_registered_test_data(dimensions, n_folds=None, keep_fold=None):
    """
    Return np.array of the nifti dataset
    """
    global N_SUBJECTS

    MODALITIES = ['3D_EPI_1x2_Reg', '3D_EPI_1x3_Reg', '3D_EPI_2x2_Reg']
    data_dict = get_data_dict()

    single_fold = n_folds is not None and keep_fold is not None
    if single_fold:
        logging.info(f'Only keeping subjects from fold {keep_fold}')
        kf = KFold(n_folds, shuffle=True, random_state=42)
        for fold_i, idxs in enumerate(kf.split(range(N_SUBJECTS))):
            if fold_i == keep_fold:
                train_i, test_i = idxs
                keep_idxs = test_i

    to_stack = []
    vol_paths = []
    for m in MODALITIES:
        arr, paths = _get_niftis(dimensions, m, data_dict)
        # 2 - (13312, 384, 312)
        # 3 - (52, 384, 312, 256)

        if single_fold:
            for i in keep_idxs:
                if dimensions == 2:
                    a = arr[i * 256: i * 256 + 256]
                elif dimensions == 3:
                    a = np.expand_dims(arr[i], axis=0) # (384,312,256) -> (1,384,312,256)

                to_stack.append(a)
                vol_paths.append(paths[i])
        else:
            to_stack.append(arr)
            vol_paths.extend(paths)

    X_test = np.vstack(to_stack)

    _test_data_shape(dimensions, X_test)

    return X_test, vol_paths


def _test_data_shape(dimensions, data):
    error_msg = 'Expected shape: {}, Given shape: {}'
    if dimensions == 2:
        expected_shape = (384, 312)
        assert data.shape[1:] == expected_shape, error_msg.format(expected_shape, data.shape[1:])
    elif dimensions == 3:
        expected_shape = (384, 312, 256)
        assert data.shape[1:] == expected_shape, error_msg.format(expected_shape, data.shape[1:])
    

def get_masks():
    '''
    Returns dictionary of
        {subj_id:
            { mask1: data,
              mask2: data,
              mask3: data
            }
        }
    '''
    MODALITIES = ['3D_T1_Reg_pve_2', 'vein_mask', 'probability_map']
    data_dict = get_data_dict()

    result = {}
    for m in MODALITIES:
        arr, paths = _get_niftis(3, m, data_dict)

        for mask, path in zip(arr, paths):
            subj_id = path.split('/')[6]
            mask_pp = _process_mask(mask, m)

            if subj_id not in result:
                result[subj_id] = {m: mask_pp}
            else:
                result[subj_id][m] = mask_pp

    return result

def _process_mask(mask, mask_type):
    wm_threshold = 0.0
    vein_threshold = 0.0
    lesion_threshold = 0.6

    if mask_type == '3D_T1_Reg_pve_2':
        mask = np.array(mask > wm_threshold, dtype=np.uint8)
    elif mask_type == 'vein_mask':
        mask = np.array(mask > vein_threshold, dtype=np.uint8)
    elif mask_type == 'probability_map':
        mask = np.array(mask > lesion_threshold, dtype=np.uint8)
    else:
        logging.info('Unknown mask type: {}'.format(mask_type))

    return mask

def get_fold_test_set(X_slices, y_slices, slc_paths, train_idx, test_idx):
    train_X = []
    train_y = []
    train_paths = []
    for i in train_idx:
        train_X.append(X_slices[i * 256: i * 256 + 256])
        train_y.append(y_slices[i * 256: i * 256 + 256])
        train_paths.extend(slc_paths[i * 256: i * 256 + 256])

    test_X = []
    test_y = []
    test_paths = []
    for j in test_idx:
        test_X.append(X_slices[j * 256: j * 256 + 256])
        test_y.append(y_slices[j * 256: j * 256 + 256])
        test_paths.extend(slc_paths[j * 256: j * 256 + 256])

    train_X = np.vstack(train_X)
    train_y = np.vstack(train_y)
    test_X = np.vstack(test_X)
    test_y = np.vstack(test_y)

    return test_X, test_y, test_paths

def train_leave_one_out_2D(X, y, paths, seed=24):
    """
    Split given lists by number of subjects into 80:20 ratio
    """
    global N_SUBJECTS
    split_i = int(N_SUBJECTS * 0.8)
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

def train_leave_one_out_3D(X, y, paths, seed=24):
    global N_SUBJECTS
    split_i = int(N_SUBJECTS * 0.8)
    shuffle_i = np.random.RandomState(seed=seed).permutation(len(X))
    train_subj_i, test_subj_i = shuffle_i[:split_i], shuffle_i[split_i:]

    train_paths = [paths[idx] for idx in train_subj_i]
    test_paths  = [paths[idx] for idx in test_subj_i]

    train = X[train_subj_i], y[train_subj_i], train_paths
    test = X[test_subj_i], y[test_subj_i], test_paths

    return train, test

def get_data_dict(json_file_path=DATA_FILE):
    """
    Create dict for 
        {subj_id:
            {modality: 
                [dicom_paths]
            }
        }
    """
    data_dict = {}
    DICOMS_PATH = '/home/quahb/caipi_denoising/data/source_dicoms/{}/{}/'
    NIFTI_PATH  = '/home/quahb/caipi_denoising/data/source_niftis/{}/{}.nii.gz'
    
    with open(json_file_path) as json_file:
        data_json = json.load(json_file)

        for subj in data_json['subjects']:
            data_dict[subj] = {}

            for modal in data_json['dicom_modalities']:
                path = DICOMS_PATH.format(subj, modal)
                subj_dicoms = os.listdir(path)
                for i in range(len(subj_dicoms)): subj_dicoms[i] = path + subj_dicoms[i]
                data_dict[subj][modal] = subj_dicoms

            for modal in data_json['nifti_modalities']:
                path = NIFTI_PATH.format(subj, modal)
                data_dict[subj][modal] = path

            for modal in data_json['mask_modalities']:
                path = NIFTI_PATH.format(subj, modal)
                data_dict[subj][modal] = path

    return data_dict

def _get_dicoms(dimensions, modality, data_dict):
    """
    Return np.array of all dicom slices for a single modality. 
    Dicom slices are sorted subject-wise.
    """
    all_images = []
    all_paths  = []
    for subj in tqdm(data_dict.keys(), ncols=80, desc=modality):
        subj_slices_path = data_dict[subj][modality]
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

        if dimensions == 2:
            all_images.extend([s for s in subj_slices])
            all_paths.extend(subj_slices_path)
        elif dimensions == 3:
            all_images.append([s for s in subj_slices])
            all_paths.append(subj_slices_path)
    
    all_images = np.stack(all_images, axis=0)

    if dimensions == 3:
        all_images = np.moveaxis(all_images, 1, -1)

    return all_images, all_paths

def _get_niftis(dimensions, modality, data_dict):
    all_vols = []
    all_paths  = []
    
    for subj in tqdm(data_dict.keys(), ncols=80, desc=modality):
        vol_path = data_dict[subj][modality]
        vol_data = nib.load(vol_path).get_fdata()
        vol_data = np.moveaxis(vol_data, 0, 1) # (312, 384, 256) -> (384, 312, 256)
        vol_data = np.flip(vol_data, axis=2)

        all_vols.append(vol_data)
        all_paths.append(vol_path)

    if dimensions == 2:
        all_vols = [ np.moveaxis(vol, -1, 0) for vol in all_vols ]
        all_vols = np.vstack(all_vols)
    elif dimensions == 3:
        all_vols = np.stack(all_vols)

    return all_vols, all_paths

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
    DICOMS_PATH = '/home/quahb/caipi_denoising/data/dicoms/{}/{}/'
    NIFTI_PATH  = '/home/quahb/caipi_denoising/data/niftis/{}/{}.nii.gz'
    
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


def get_dicoms(dimensions, modality, data_dict):
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

        if 'pha' in modality:
            subj_slices = [s.pixel_array * s.RescaleSlope + s.RescaleIntercept for s in subj_slices]
        else:
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


def get_raw_data(dataset_type, modalities):
    RAW_DATA_PATH = '/home/quahb/caipi_denoising/data/raw_data/'
    TRAIN_NAME = '3D_T2STAR_segEPI'
    TEST_NAME  = 'CAIPI'

    res_data = []
    res_names = []

    if type(modalities) is not list:
        modalities = [modalities]

    for m in modalities:
        folder_path = os.path.join(RAW_DATA_PATH, m)
        file_names  = sorted(os.listdir(folder_path))

        if dataset_type == 'train':
            file_names = [ f for f in file_names if TRAIN_NAME in f ]
        elif dataset_type == 'test':
            file_names = [ f for f in file_names if TEST_NAME in f ]

        base_names = [ f.split('.')[0] for f in file_names ]
        res_names.extend(base_names)

        file_names = [ os.path.join(folder_path, f) for f in file_names ]
        for f in file_names:
            res_data.append(np.load(f))

    res_data = np.stack(res_data)

    return res_data, res_names
    

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
            mask_pp = _threshold_mask(mask, m)

            if subj_id not in result:
                result[subj_id] = {m: mask_pp}
            else:
                result[subj_id][m] = mask_pp

    return result


def _threshold_mask(mask, mask_type):
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

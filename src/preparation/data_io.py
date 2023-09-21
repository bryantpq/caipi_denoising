import logging
import multiprocessing as mp
import nibabel as nib
import numpy as np
import os
import pdb
from tqdm import tqdm

from preparation.preprocessing_pipeline import rescale_magnitude


def load_dataset(data_folder, dimensions, data_format, return_names=False, subset=None):
    files = sorted(os.listdir(data_folder))
    files = [ f for f in files if os.path.isfile(os.path.join(data_folder, f)) ]

    if subset is not None: # take first n subjects only
        files = files[:subset]

    data = []
    pbar = tqdm(files, ncols=110)
    for f in pbar:
        pbar.set_description(f'{f}')

        file_ext = '.'.join(f.split('.')[1:])

        fpath = os.path.join(data_folder, f)
        if file_ext == 'npy':
            data.append( np.load(fpath) )
        elif file_ext == 'nii.gz':
            data.append( np.array(nib.load(fpath).dataobj) )
        else:
            raise ValueError(f'Detected bad file extension: {file_ext}')

    if dimensions == 2:
        if data_format == 'full': # input dims eg. [384, 384, 256]
            data = [ np.moveaxis(d, -1, 0) for d in data ]
            data = np.vstack(data)
        elif data_format == 'patches': # input dims eg. [1000, 256, 256]
            data = np.vstack(data)

        data = np.expand_dims(data, axis=-1)
        assert data.ndim == 4, '2D Data for training should be 4 dimensional: [num_samples, dim1, dim2, 1]'

    elif dimensions == 3:
        if data_format == 'full': # input dims eg. [384, 384, 256]
            data = np.stack(data)
        elif data_format == 'patches': # input dims eg. [9, 256, 256, 256]
            data = np.vstack(data)

        data = np.expand_dims(data, axis=-1)
        assert data.ndim == 5, '3D Data for training should be 5 dimensional: [num_samples, dim1, dim2, dim3, 1]'

    if return_names:
        return data, files
    else:
        return data

def load_raw_niftis(load_path, load_modalities, rescale_combine_mag_phase=True):
    '''
    Given a path to a folder of subjects, load given modalities within each subject.
    '''
    res = {}
    subj_ids = sorted(os.listdir(load_path))

    pbar = tqdm(subj_ids, ncols=100)
    for subj in pbar:
        pbar.set_description(f'Loading nifti {subj}')
        files = os.listdir(os.path.join(load_path, subj))
        file_paths = [ os.path.join(load_path, subj, f) for f in files ]

        res[subj] = {}
        for lm in load_modalities:
            for fp in file_paths:
                fname = fp.split('/')[-1].split('.')[0]
                if lm == fname: # e.g. CAIPI1x2 in ../data/dataset/subj_id/CAIPI1x2.nii.gz
                    tmp = np.array(nib.load(fp).dataobj)
                    res[subj][lm] = tmp

    if rescale_combine_mag_phase:
        res = _rescale_combine_mag_phase(res)

    return res, len(subj_ids)

def _rescale_combine_mag_phase(data_dict, remove_noncomplex=True):
    pbar = tqdm(list(data_dict.keys()), ncols=100, total=len(data_dict.keys()))
    for subj in pbar:
        pbar.set_description(f'Reconstructing complex data for {subj}')
        subj_modalities = data_dict[subj].keys()
        to_convert = [ sm for sm in subj_modalities if sm + '_pha' in subj_modalities]

        for m in to_convert: 
            # assign complex data to magnitude key-val, remove phase k-v
            mag, pha = data_dict[subj][m], data_dict[subj][m + '_pha']
            data_dict[subj][m] = magphase2complex(mag, pha)
            del data_dict[subj][m + '_pha']

        if remove_noncomplex:
            # remove modalities that were not reconstructed to complex
            # eg, if they have mag but no phase, then subject will have empty dict
            for sm in list(data_dict[subj].keys()):
                if sm not in to_convert: 
                    del data_dict[subj][sm]

    return data_dict

def magphase2complex(mag, pha):
    assert mag.shape == pha.shape

    re_mag = rescale_magnitude(mag, 0, 1)
    re_pha = rescale_magnitude(pha, -np.pi, np.pi)

    complex_image = np.multiply(re_mag, np.exp(1j * re_pha)).astype('complex64')

    return complex_image

def unpack_data_dict(data_dict):
    data, names = [], []

    for subj in data_dict.keys():
        for modal in data_dict[subj].keys():
            data.append(data_dict[subj][modal])
            names.append(f'{subj}_{modal}')

    return data, names

def write_data(data, filename, save_dtype):
    create_folders('/'.join(filename.split('/')[:-1]))
    data = data.astype(save_dtype)

    np.save(filename, data)

def create_folders(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def threshold_mask(mask, mask_type):
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

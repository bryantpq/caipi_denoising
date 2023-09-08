import argparse
import logging
import nibabel as nib
import numpy as np
import pdb
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import shutil
import yaml

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from preparation.data_io import load_niftis, unpack_data_dict, create_folders
from preparation.extract_patches import extract_patches
from preparation.preprocessing_pipeline import preprocess_data
from utils.create_logger import create_logger

def main():
    parser = create_parser()
    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    config_file = args.config.name.split('/')[-1]
    config_name = config_file.split('.')[0]
    create_logger(config_name, config['logging_level'])
    
    logging.info(config)
    logging.info('')

    acceleration = config['output_folder'].split('/')[-2]
    combine_mag_phase = config['combine_mag_phase']
    output_folder = config['output_folder']
    save_format = config['save_format']
    test_size = config['test_size']

    assert acceleration in ['unaccelerated', 'accelerated']
    assert save_format in ['npy', 'python', 'numpy', 'nifti', 'nii']

    if acceleration == 'unaccelerated':
        load_modalities = config['unaccelerated_modalities']
    elif acceleration == 'accelerated':
        load_modalities = config['accelerated_modalities']

    data_dict, n_subjs = load_niftis(
            config['input_folder'], 
            load_modalities, 
            combine_mag_phase=combine_mag_phase
    )

    assert len(data_dict.keys()) == n_subjs
    idxs = list(range(n_subjs))
    
    if test_size == 0:
        train_idxs, test_idxs = idxs, []
    elif test_size == 1:
        train_idxs, test_idxs = [], idxs
    else:
        train_idxs, test_idxs = train_test_split(idxs, test_size=test_size,
                random_state=config['split_seed'])

    train_dict, test_dict = split_data(data_dict, train_idxs, test_idxs)
    del data_dict

    if acceleration == 'unaccelerated':
        data, names = unpack_data_dict(train_dict)
        create_folders(os.path.join(output_folder, 'images'))
        create_folders(os.path.join(output_folder, 'labels'))
        pbar = tqdm(enumerate(zip(data, names)), ncols=100, total=len(names))
        for i, (d, n) in pbar:
            pbar.set_description(f'Preprocessing {n}')
            image = d
            label = np.copy(image)

            processed_image = preprocess_data(
                    image, config['preprocessing_params'], config['image_steps'], subj_i=i
            )
            processed_label = preprocess_data(
                    label, config['preprocessing_params'], config['label_steps'], subj_i=i
            )

            if config['extract_patches']:
                ep_params = config['extract_patches_params']
                processed_image = extract_patches(processed_image, config['dimensions'], **ep_params)
                processed_label = extract_patches(processed_label, config['dimensions'], **ep_params)

            image_fname = os.path.join(output_folder, 'images', n)
            label_fname = os.path.join(output_folder, 'labels', n)
            if save_format in ['npy', 'numpy', 'python']:
                np.save(image_fname, processed_image)
                np.save(label_fname, processed_label)
            elif save_format in ['nii', 'nifti']:
                nii_image = nib.Nifti1Image(processed_image, affine=np.eye(4))
                nii_label = nib.Nifti1Image(processed_label, affine=np.eye(4))
                nib.save(nii_image, f'{image_fname}.nii.gz')
                nib.save(nii_label, f'{label_fname}.nii.gz')

    elif acceleration == 'accelerated':
        data, names = unpack_data_dict(test_dict)
        create_folders(os.path.join(output_folder, 'inputs'))
        create_folders(os.path.join(output_folder, 'outputs'))
        pbar = tqdm(zip(data, names), total=len(names), ncols=100)
        for d, n in pbar:
            pbar.set_description(f'Preprocessing {n}')
            image = d

            processed_input = preprocess_data(
                    image, config['preprocessing_params'], config['input_steps']
            )

            input_fname = os.path.join(config['output_folder'], 'inputs', n) 
            if save_format in ['npy', 'numpy', 'python']:
                np.save(input_fname, processed_input)
            elif save_format in ['nii', 'nifti']:
                nii_input = nib.Nifti1Image(processed_input, affine=np.eye(4))
                nib.save(nii_input, f'{input_fname}.nii.gz')

    logging.info(f'Completed config: {args.config.name}')

def split_data(data_dict, train_idxs, test_idxs):
    subjs = list(data_dict.keys())
    train_dict, test_dict = {}, {}

    for i in train_idxs:
        train_dict[subjs[i]] = data_dict[subjs[i]]
    for i in test_idxs:
        test_dict[subjs[i]]  = data_dict[subjs[i]]

    return train_dict, test_dict

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=argparse.FileType('r'))

    return parser

if __name__ == '__main__':
    main()

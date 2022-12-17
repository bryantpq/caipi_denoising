import argparse
import logging
import numpy as np
import pdb
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import yaml

from sklearn.model_selection import KFold, train_test_split
from preparation.gen_data import N_SUBJECTS, get_raw_data
from preparation.preprocessing_pipeline import preprocess_data
from preparation.extract_patches import extract_patches
from utils.data_io import create_folders
from utils.create_logger import create_logger


def main():
    parser = create_parser()
    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    create_logger(config['config_name'], config['logging_level'])
    
    logging.info(config)
    logging.info('')

    dataset_type = config['generate_dataset']['dataset_type']
    n_folds = config['generate_dataset']['n_folds']

    if dataset_type in ['train', 'test']:
        logging.info(f'Loading {dataset_type}ing set...')
        data, names = get_raw_data(dataset_type, config['generate_dataset']['raw_data_modalities'])
        # data.shape = [n_subj, 384, 312, 256]
        # names = [subj_id_modality, ... ] '1_01_016-V1_3D_T2STAR_segEPI'

        if n_folds in [False, None]:
            logging.info(f'Splitting {dataset_type}ing set as 1 fold...')
            idxs = list(range(N_SUBJECTS))
            keep_idxs,holdout_idxs = train_test_split(idxs, test_size=0.2, random_state=42)
            keep_data, holdout_data, keep_names, holdout_names = split_data(
                    dataset_type, data, names, keep_idxs, holdout_idxs
            )

            if dataset_type == 'train':
                process_fold(
                        dataset_type,
                        keep_data,
                        keep_names,
                        config,
                        -1
                )
            elif dataset_type == 'test':
                process_fold(
                        dataset_type,
                        holdout_data,
                        holdout_names,
                        config,
                        -1
                )

        else:
            logging.info(f'Splitting {dataset_type}ing set into {n_folds} folds...')
            kf = KFold(n_folds, shuffle=True, random_state=42)

            for fold_i, keep_holdout_idxs in enumerate(kf.split(range(N_SUBJECTS))):
                keep_idxs, holdout_idxs = keep_holdout_idxs
                keep_data, holdout_data, keep_names, holdout_names = split_data(
                        dataset_type, data, names, keep_idxs, holdout_idxs
                )

                # for every fold, we save the holdout data NOT the keep data
                # when loading for training/testing, we load the folds we want
                process_fold(
                        dataset_type,
                        holdout_data,
                        holdout_names,
                        config,
                        fold_i
                )

    elif dataset_type == 'reg_test':
        raise NotImplementedError('To implement when data is ready')

def process_fold(
        dataset_type,
        data,
        names,
        config,
        fold_i
    ):
    '''
    Given an array of data volumes and subject names with modalities, run the
    preprocessing pipeline on the data and then save the data to disk.
    '''
    if fold_i >= 0:
        logging.info(f'Processing fold {fold_i}...')
    else:
        logging.info(f'Processing whole dataset as single fold...')

    if dataset_type == 'train':
        FOLDER_PATH = '/home/quahb/caipi_denoising/data/datasets/train/{}'.format(config['config_name'])
        create_folders(os.path.join(FOLDER_PATH, 'images'))
        create_folders(os.path.join(FOLDER_PATH, 'labels'))
        images = data
        labels = np.copy(images)

        logging.info('Preprocessing images...')
        processed_images = preprocess_data(
                images,
                config['generate_dataset']['preprocessing_params'],
                config['generate_dataset']['image_steps']
        )
        logging.info('Preprocessing labels...')
        processed_labels = preprocess_data(
                labels,
                config['generate_dataset']['preprocessing_params'],
                config['generate_dataset']['label_steps']
        )
        logging.info('')

        if config['generate_dataset']['extract_patches']: # save as slices
            patches_params = config['generate_dataset']['extract_patches_params']
            logging.info('Extracting patches...')
            if fold_i >= 0:
                filename_placeholder = '{}_f' + str(fold_i)
            else:
                filename_placeholder = '{}'

            extract_patches(
                    processed_images,
                    os.path.join(FOLDER_PATH, 'images', filename_placeholder),
                    patch_size=patches_params['patch_size'],
                    extract_step=patches_params['extract_step'],
                    pad_before_ext=patches_params['pad_before_ext'],
                    pad_value=patches_params['pad_value'],
                    save_dtype=config['save_dtype'],
                    workers=config['workers']
            )
            extract_patches(
                    processed_labels,
                    os.path.join(FOLDER_PATH, 'labels', filename_placeholder),
                    patch_size=patches_params['patch_size'],
                    extract_step=patches_params['extract_step'],
                    pad_before_ext=patches_params['pad_before_ext'],
                    pad_value=patches_params['pad_value'],
                    save_dtype=config['save_dtype'],
                    workers=config['workers']
            )
        else: # save as whole volumes
            logging.info('Saving preprocessed training images and labels by subject...')
            for image, label, name in zip(processed_images, processed_labels, names):
                if fold_i >= 0:
                    filename = name + '_f' + str(fold_i)
                else:
                    filename = name
                image_filename = os.path.join(FOLDER_PATH, 'images', filename)
                label_filename = os.path.join(FOLDER_PATH, 'labels', filename)
                np.save(image_filename, image)
                np.save(label_filename, label)

    elif dataset_type == 'test':
        FOLDER_PATH = '/home/quahb/caipi_denoising/data/datasets/test/{}'.format(config['config_name'])
        create_folders(os.path.join(FOLDER_PATH, 'inputs'))
        create_folders(os.path.join(FOLDER_PATH, 'outputs'))
        inputs = data

        logging.info('Preprocessing inputs...')
        processed_inputs = preprocess_data(
                inputs,
                config['generate_dataset']['preprocessing_params'],
                config['generate_dataset']['input_steps']
        )
        logging.info('')

        logging.info('Saving preprocessed testing inputs...')
        for image, name in zip(processed_inputs, names):
            if fold_i >= 0:
                filename = name + '_f' + str(fold_i)
            else:
                filename = name
            image_filename = os.path.join(FOLDER_PATH, 'inputs', filename)
            np.save(image_filename, image)
    
def split_data(dataset_type, data, names, keep_idxs, holdout_idxs):
    keep_stack, holdout_stack = [], []
    keep_names, holdout_names = [], []

    if dataset_type == 'train':
        for i in keep_idxs:
            keep_stack.append(data[i])
            keep_names.append(names[i])
        for j in holdout_idxs:
            holdout_stack.append(data[j])
            holdout_names.append(names[j])
    elif dataset_type == 'test':
        N_CAIPI_MODALITIES = 3
        # names are sorted by: [subj_1_CAIPI2, subj_1_CAIPI3, subj_1_CAIPI4, ...]
        for i in keep_idxs:
            i = i * N_CAIPI_MODALITIES
            keep_stack.extend([ data[i], data[i + 1], data[i + 2] ])
            keep_names.extend([ names[i], names[i + 1], names[i + 2] ])
        for j in holdout_idxs:
            j = j * N_CAIPI_MODALITIES
            holdout_stack.extend([ data[j], data[j + 1], data[j + 2] ])
            holdout_names.extend([ names[j], names[j + 1], names[j + 2] ])

    logging.info('Keep set:')
    logging.info(keep_names)
    logging.info('Holdout set:')
    logging.info(holdout_names)

    keep_stack = np.stack(keep_stack)
    holdout_stack = np.stack(holdout_stack)

    return keep_stack, holdout_stack, keep_names, holdout_names

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=argparse.FileType('r'))

    return parser

if __name__ == '__main__':
    main()

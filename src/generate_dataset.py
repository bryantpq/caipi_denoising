import argparse
import logging
import numpy as np
import pdb
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import yaml

from sklearn.model_selection import KFold
from preparation.gen_data import N_SUBJECTS
from preparation.gen_data import get_train_data, get_test_data, get_registered_test_data, get_fold_test_set
from preparation.preprocessing_pipeline import preprocess_slices
from preparation.extract_patches import extract_patches
from utils.data_io import write_slices
from utils.create_logger import create_logger


def main():
    parser = create_parser()
    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    create_logger(config['config_name'], config['logging_level'])
    
    logging.info(config)
    logging.info('')

    NIFTI_PATHS = False
    if config['generate_dataset']['train_or_test_set'] == 'train':
        logging.info('Loading training set...')
        n_folds = config['generate_dataset']['n_folds']
        if n_folds in [False, None]:
            logging.info('Loading whole training dataset...')
            X_slices, y_slices, slc_paths = get_train_data(config['dimensions'])
        else:
            logging.info(f'Splitting dataset into {n_folds} folds')
            X_slices, y_slices, slc_paths = get_train_data(config['dimensions'])
            kf = KFold(n_folds, shuffle=True, random_state=42)

            for fold_i, train_test_idxs in enumerate(kf.split(range(N_SUBJECTS))):
                train_idxs, test_idxs = train_test_idxs
                cur_test_X, cur_test_y, cur_slc_paths = get_fold_test_set(X_slices, y_slices, slc_paths, 
                        train_idxs, test_idxs)
                test_subjs = [ p.split('/')[6] for p in cur_slc_paths[::256] ]
                logging.info(f'Testing set for fold {fold_i}:')
                logging.info(test_subjs)

                process_fold(config,
                             fold_i,
                             cur_test_X,
                             cur_test_y,
                             cur_slc_paths,
                             NIFTI_PATHS)

            return

    elif config['generate_dataset']['train_or_test_set'] == 'test':
        logging.info('Loading testing set...')
        X_slices, slc_paths = get_test_data(config['dimensions'])
        y_slices = np.copy(X_slices)

    elif config['generate_dataset']['train_or_test_set'] == 'reg_test':
        logging.info('Loading coregistered testing set...')
        X_slices, slc_paths = get_registered_test_data(config['dimensions'])
        y_slices = np.copy(X_slices)

        NIFTI_PATHS = True

    process_fold(config, -1, X_slices, y_slices, slc_paths, NIFTI_PATHS)

def process_fold(
            config, 
            fold_i, 
            X_slices, 
            y_slices,
            slc_paths,
            NIFTI_PATHS
    ):
    if fold_i >= 0: 
        logging.info(f'Processing fold {fold_i}...')
    else:
        logging.info('Processing single fold...')

    logging.info('X_slices.shape: {}, y_slices.shape: {}'.format(X_slices.shape, y_slices.shape))
    
    logging.info('Preprocessing X slices...')
    X_slices = preprocess_slices(X_slices,
                                 config['dimensions'],
                                 config['generate_dataset']['preprocessing_params'],
                                 steps=config['generate_dataset']['X_steps'])
    logging.info('Preprocessing y slices...')
    y_slices = preprocess_slices(y_slices,
                                 config['dimensions'],
                                 config['generate_dataset']['preprocessing_params'],
                                 steps=config['generate_dataset']['y_steps'])
    logging.info('')

    if config['generate_dataset']['extract_patches']:
        patches_params = config['generate_dataset']['extract_patches_params']
        logging.info('Extracting then saving patches...')

        if fold_i >= 0:
            x_name = f'X_f{fold_i}'
            y_name = f'y_f{fold_i}'
        else:
            x_name = 'X'
            y_name = 'y'
            
        logging.info('Processing X...')
        extract_patches(X_slices,
                        x_name,
                        save_path=config['data_folder'],
                        dimensions=config['dimensions'],
                        patch_size=patches_params['patch_size'],
                        extract_step=patches_params['extract_step'],
                        pad_before_ext=patches_params['pad_before_ext'],
                        pad_value=patches_params['pad_value'],
                        save_dtype=config['save_dtype'],
                        workers=config['workers'])
        logging.info('')

        logging.info('Processing y...')
        extract_patches(y_slices,
                        y_name,
                        save_path=config['data_folder'],
                        dimensions=config['dimensions'],
                        patch_size=patches_params['patch_size'],
                        extract_step=patches_params['extract_step'],
                        pad_before_ext=patches_params['pad_before_ext'],
                        pad_value=patches_params['pad_value'],
                        save_dtype=config['save_dtype'],
                        workers=config['workers'])
        logging.info('')

    else:
        if config['dimensions'] == 2:
            logging.info('Saving slices')

            write_slices(X_slices, 'X', config['data_folder'], config['generate_dataset']['save_dtype'])
            write_slices(y_slices, 'y', config['data_folder'], config['generate_dataset']['save_dtype'])
        elif config['dimensions'] == 3:
            logging.info('Saving whole volumes...')

            n_subj = len(X_slices)
            for subj_i in range(n_subj):
                cur_X = X_slices[subj_i]
                cur_y = y_slices[subj_i]

                if NIFTI_PATHS:
                    cur_subj_id = slc_paths[subj_i].split('/')[6]
                    cur_modality = slc_paths[subj_i].split('/')[7].split('.')[0]
                else: # dicom paths
                    cur_subj_id = slc_paths[subj_i][0].split('/')[6]
                    cur_modality = slc_paths[subj_i][0].split('/')[7]
                
                X_name = f'{cur_subj_id}_{cur_modality}_X'
                y_name = f'{cur_subj_id}_{cur_modality}_y'
                write_slices(cur_X, X_name, config['data_folder'], config['save_dtype'])
                write_slices(cur_y, y_name, config['data_folder'], config['save_dtype'])

    logging.info('Generating dataset complete for config: {}'.format(config['config_name']))

    
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=argparse.FileType('r'))

    return parser

if __name__ == '__main__':
    main()

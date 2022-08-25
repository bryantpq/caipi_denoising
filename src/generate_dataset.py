import argparse
import logging
import numpy as np
import pdb
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import yaml

from preparation.gen_data import get_train_data, get_test_data
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


    if config['generate_dataset']['train_or_test_set'] == 'train':
        logging.info('Loading training set...')
        if config['generate_dataset']['train_leave_one_out']:
            logging.info('Leaving 1/5 folds out for testing')
            logging.info('Loading 4/5 folds for training...')
            X_slices, y_slices, slc_paths = get_train_data(config['dimensions'], train_loo='train')
        else:
            logging.info('Loading whole training dataset...')
            X_slices, y_slices, slc_paths = get_train_data(config['dimensions'], train_loo=False)

    elif config['generate_dataset']['train_or_test_set'] == 'test':
        logging.info('Loading testing set...')
        X_slices, slc_paths = get_test_data(config['dimensions'])
        y_slices = np.copy(X_slices)

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

        logging.info('Processing X...')
        extract_patches(X_slices, 'X',
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
        extract_patches(y_slices, 'y',
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
        assert config['dimensions'] == 3, 'Saving volumes only configured for 3D.'
        logging.info('Saving whole volumes...')

        n_subj = len(X_slices)
        for subj_i in range(n_subj):
            cur_X = X_slices[subj_i]
            cur_y = y_slices[subj_i]
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

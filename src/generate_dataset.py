import argparse
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import yaml

from preparation.gen_data import get_train_data
from preparation.preprocessing_pipeline import preprocess_slices
from preparation.extract_patches import extract_patches
from utils.data_io import write_slices


def main():
    parser = create_parser()
    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    
    print(config)
    print()
    if config['generate_dataset']['median_slices']:
        print('Gathering median slices...')
    else:
        print('Gathering all slices...')
    X_slices, y_slices = get_train_data(median_slices=config['generate_dataset']['median_slices'])
    print('X_slices.shape: {}, y_slices.shape: {}'.format(X_slices.shape, y_slices.shape))
    
    print('Preprocessing X slices...')
    X_slices_pp = preprocess_slices(X_slices,
                                   config['generate_dataset']['preprocessing_params'],
                                   steps=config['generate_dataset']['X_steps'])
    print('Preprocessing y slices...')
    y_slices_pp = preprocess_slices(y_slices,
                                   config['generate_dataset']['preprocessing_params'],
                                   steps=config['generate_dataset']['y_steps'])

    if config['generate_dataset']['extract_patches']:
        patches_params = config['generate_dataset']['extract_patches_params']
        print('Extracting then saving patches...')
        extract_patches(X_slices_pp, 'X',
                        save_path=config['data_folder'],
                        patch_size=patches_params['patch_size'],
                        extract_step=patches_params['extract_step'],
                        pad_before_ext=patches_params['pad_before_ext'],
                        pad_value=patches_params['pad_value'],
                        save_dtype=config['generate_dataset']['save_dtype'],
                        workers=config['workers'])
        extract_patches(y_slices_pp, 'y',
                        save_path=config['data_folder'],
                        patch_size=patches_params['patch_size'],
                        extract_step=patches_params['extract_step'],
                        pad_before_ext=patches_params['pad_before_ext'],
                        pad_value=patches_params['pad_value'],
                        save_dtype=config['generate_dataset']['save_dtype'],
                        workers=config['workers'])

    else:
        print('Saving slices...')
        write_slices(X_slices_pp, 'X', config['data_folder'], config['generate_dataset']['save_dtype'])
        write_slices(y_slices_pp, 'y', config['data_folder'], config['generate_dataset']['save_dtype'])

    print('Generating dataset complete for config: {}'.format(config['config_name']))

    
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=argparse.FileType('r'))

    return parser

if __name__ == '__main__':
    main()

import argparse
import logging
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pdb
import tensorflow as tf
from tqdm import tqdm
import yaml

from modeling.get_model import get_model
from preparation.prepare_tf_dataset import np_to_tfdataset
from utils.data_io import load_dataset, write_data
from utils.create_logger import create_logger

from patchify import patchify, unpatchify

def main():
    parser = create_parser()
    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    create_logger(config['config_name'], config['logging_level'])

    logging.info(config)
    logging.info('')
    logging.info('Loading testing dataset from: {}'.format(config['results_folder']))

    inputs_path = os.path.join(config['results_folder'], 'inputs')
    inputs, fnames = load_dataset(inputs_path, load_folds=config['test_fold'], postprocess_mode='test')
    logging.info('')
    logging.info('Creating model...')

    strategy = tf.distribute.MirroredStrategy(devices=config['gpus'])
    with strategy.scope():
        model = get_model(
                model_type=config['model_type'], 
                loss_function=config['loss_function'],
                input_shape=config['input_shape'],
                load_model_path=config['predict_test']['load_model_path'],
                init_alpha=[1, 1],
                noise_window_size=config['train_network']['noise_window_size'])

    if config['predict_test']['extract_patches']:
        patch_size = config['input_shape'][1:3]
        extract_step = config['predict_test']['extract_step']
        logging.info('Model prediction using patches...')
    else:
        logging.info('Model prediction using full slices...')
    logging.info('')

    # iterate over testing set and run prediction on each subject
    for subj_i, (data, input_name) in enumerate(zip(inputs, fnames)):
        # data.shape (384,384,256)
        logging.info(f'{subj_i + 1}/{len(fnames)} - Running prediction on {input_name} ...')

        if config['predict_test']['extract_patches']:
            # extract patches for every slice in subject
            patches_stack = []
            for i in range(data.shape[-1]): 
                slc = np.copy(data[:,:,i])
                patches = patchify(slc, patch_size, step=extract_step)
                patchify_shape = patches.shape[:2]
                patches = patches.reshape(-1, *patch_size)
                patches = np.expand_dims(patches, axis=3)
                patches_stack.append(patches)
            patches_stack = np.vstack(patches_stack) #(n_patches,patch_size,patch_size,1)

            logging.info(f'Collected patches for prediction: {patches_stack.shape}')
            
            # run prediction
            logging.info('Running prediction...')
            patches_stack = np_to_tfdataset(patches_stack, complex_split=config['complex_split'])
            patches = model.predict(
                    patches_stack,
                    verbose=1,
                    batch_size=32)
            logging.info('Done!')
            patches = np.squeeze(patches)
            if config['complex_split']: # reconstruct the complex number from real imag
                res = np.zeros(patches.shape[:-1], dtype=config['save_dtype'])
                res = patches[:,:,:,0] + 1j * patches[:,:,:,1]
                patches = res

            # reconstruct slices from patches
            subj_slices= []
            n_patches_per_slice = len(patches) // data.shape[-1] # should be 2304 / 256 = 9
            patches = patches.reshape(data.shape[-1], *patchify_shape, *patch_size)
            # patches.shape = (256, 3, 3, 256, 256)
            for slc_i in range(patches.shape[0]):
                slc = unpatchify(patches[slc_i], data.shape[:-1])
                subj_slices.append(slc)
            denoised_subject = np.stack(subj_slices)
            output = np.moveaxis(denoised_subject, 0, -1)
        else:
            logging.info('Running prediction on full slices...')
            data = np.moveaxis(data, -1, 0)
            data = np.expand_dims(data, axis=3)
            data = np_to_tfdataset(data, complex_split=config['complex_split'])
            data = model.predict(
                    data,
                    verbose=1,
                    batch_size=32)
            data = np.squeeze(data)
            if config['complex_split']:
                # data.shape = [256, 384, 384, 2]
                res = np.zeros(data.shape[:-1], dtype=config['save_dtype'])
                res = data[:,:,:,0] + 1j * data[:,:,:,1]
                data = res
            output = np.moveaxis(data, 0, -1)

        # save data
        output_name = os.path.join(inputs_path, input_name)
        output_name = output_name.replace('inputs', 'outputs')
        write_data(output, output_name, config['save_dtype'])

        logging.info(f'Saving data to {output_name}')
        logging.info('------------------------------')

    logging.info('Prediction complete for config: {}'.format(config['config_name']))

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=argparse.FileType('r'))

    return parser

if __name__ == '__main__':
    main()

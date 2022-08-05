import argparse
import logging
import numpy as np
import os
import tensorflow as tf
import yaml

from evaluation.report_metrics import report_metrics
from modeling.get_model import get_model
from preparation.gen_data import get_test_data, get_train_data
from preparation.prepare_tf_dataset import np_to_tfdataset
from preparation.preprocessing_pipeline import preprocess_slices
from utils.data_io import write_slices
from utils.create_logger import create_logger

from patchify import patchify, unpatchify


def main():
    parser = create_parser()
    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    create_logger(config['config_name'])

    logging.info(config)
    logging.info('')

    if config['predict_test']['train_or_test_set'] == 'test':
        logging.info('Loading testing data...')
        X_test, test_paths = get_test_data()
    elif config['predict_test']['train_or_test_set'] == 'train': 
        logging.info('Loading training data...')
        X_test, _ = get_train_data()
        del _
    
    if config['predict_test']['shuffle']:
        logging.info('Shuffling slices for prediction...')
        shuffle_i = np.random.RandomState(seed=config['predict_test']['seed']).permutation(len(X_test))
        X_test = X_test[shuffle_i]
    else:
        logging.info('Not shuffling slices...')

    logging.info(f'X_test.shape: {X_test.shape}')
    
    logging.info('Preprocessing X_test')
    X_test = preprocess_slices(X_test,
                               config['predict_test']['preprocessing_params'],
                               steps=config['predict_test']['X_steps'])
    logging.info('X_test.shape: {}'.format(X_test.shape))
    
    predict_n_slices = config['predict_test']['predict_n_slices']
    if predict_n_slices is None:
        logging.info('    Prediction will run on all {} slices.'.format(len(X_test)))
    else:
        logging.info('    Prediction will run on first {} slices.'.format(predict_n_slices))
        X_test = X_test[:predict_n_slices]

    logging.info('')
    logging.info('Creating model...')
    strategy = tf.distribute.MirroredStrategy(devices=config['gpus'])
    with strategy.scope():
        model = get_model(model_type=config['predict_test']['model_type'], 
                          input_shape=config['predict_test']['input_shape'],
                          load_model_path=config['predict_test']['load_model_path'])
    logging.info(model.summary())
    
    if config['predict_test']['extract_patches']:
        logging.info('Prediction on patches of slices...')
        patch_size = config['predict_test']['input_shape'][1:3]
        extract_step = config['predict_test']['extract_step']
        
        # TODO
        # Refactor this
        if config['predict_test']['recon_patches']:
            logging.info('Predicting on all patches then reconstructing...')
            y_test = []
            for i in range(len(X_test)):
                logging.info( '    Slice {} / {}'.format(i + 1, len(X_test)) )
                slc = X_test[i][:, :, 0]

                patches = patchify(slc, patch_size, step=extract_step)
                patchify_shape = patches.shape[:2]
                patches = patches.reshape(-1, *patch_size)
                patches = np.expand_dims(patches, axis=3)

                patches = np_to_tfdataset(patches)
                patches = model.predict(patches,
                                        verbose=1,
                                        batch_size=30)
                patches = patches[:,:,:,0]

                patches = patches.reshape(*patchify_shape, *patch_size)
                res_slc = unpatchify(patches, slc.shape)
                y_test.append(res_slc)

            y_test = np.array(y_test)
        else:
            predict_patch_i = len(patches) // 2
            logging.info(f'Predicting {predict_patch_i}-th patch of the slices. No reconstruction...')
            new_X, y_test = [], []
            for i in range(len(X_test)):
                logging.info( '    Slice {} / {}'.format(i + 1, len(X_test)) )
                slc = X_test[i][:, :, 0]
                patches = patchify(slc, patch_size, step=1)
                patches = patches.reshape(-1, *patch_size)

                patch = patches[predict_patch_i]
                new_X.append(patch)
                patch = np.expand_dims(patch, axis=2)
                patch = np_to_tfdataset(patch)
                res = model.predict(patch,
                                    verbose=1,
                                    batch_size=1)
                res = res[:,:,:,0]
                y_test.append(res)

            X_test = np.array(new_X)
            y_test = np.array(y_test)
    else:
        logging.info('Prediction on full slices ...')
        X_test_tf = np_to_tfdataset(X_test)
        y_test = model.predict(X_test_tf,
                               verbose=1,
                               batch_size=30)

    if len(X_test.shape) == 4: X_test = X_test[:,:,:,0]

    logging.info('Completed prediction ...')
    logging.info('    Results shape: X_test {}, y_test {}'.format(X_test.shape, y_test.shape))
    write_slices(X_test, 'X', config['results_folder'], config['predict_test']['save_dtype'])
    write_slices(y_test, 'y', config['results_folder'], config['predict_test']['save_dtype'])

    logging.info('Prediction complete for config: {}'.format(config['config_name']))
    logging.info('Results saved at {}'.format(config['results_folder']))

    #logging.info('Generating metrics...')
    #report_metrics(X_test, y_test)

    
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=argparse.FileType('r'))

    return parser

if __name__ == '__main__':
    main()

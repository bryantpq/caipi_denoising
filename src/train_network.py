import argparse
import logging
import numpy as np
import os
import pdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import yaml

from modeling.get_model import get_model
from modeling.callbacks import get_training_cb
from preparation.prepare_tf_dataset import np_to_tfdataset
from utils.data_io import load_dataset
from utils.create_logger import create_logger


def main():
    parser = create_parser()
    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    create_logger(config['config_name'], config['logging_level'])

    logging.info(config)
    logging.info('')
    logging.info('Loading training dataset from: {}'.format(config['data_folder']))

    if config['test_fold'] not in [None, False] and config['n_folds'] not in [None, False]:
        load_folds = set(range(config['n_folds'])) - {config['test_fold']}
        logging.info(f'Loading folds: {load_folds}')
    else:
        load_folds = None

    images_path = os.path.join(config['data_folder'], 'images')
    labels_path = os.path.join(config['data_folder'], 'labels')
    images = load_dataset(images_path, load_folds, postprocess_mode='train')
    labels = load_dataset(labels_path, load_folds, postprocess_mode='train')
    logging.debug(images.shape)
    logging.debug(labels.shape)

    if config['train_network']['train_valid_split_shuffle']:
        logging.info('Shuffling data and doing train/valid split...')
        shuffle_i = np.random.RandomState(seed=42).permutation(len(images))
        images, labels = images[shuffle_i], labels[shuffle_i]
    else:
        logging.info('Not shuffling data for train/valid split...')
    
    val_i = int( len(images) * config['train_network']['valid_split'] )
    X_train, y_train = images[:val_i], labels[:val_i]
    X_valid, y_valid = images[val_i:], labels[val_i:]

    if config['trim_batch_end']:
        train_lim = X_train.shape[0] // 32 * 32
        valid_lim = X_valid.shape[0] // 32 * 32
        X_train, y_train = X_train[:train_lim], y_train[:train_lim]
        X_valid, y_valid = X_valid[:valid_lim], y_valid[:valid_lim]

    del images, labels
    
    logging.info('Train/Valid split:')
    logging.info('{}, {}, {}, {}'.format(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape))

    train_data = np_to_tfdataset(
            X_train, y_train,
            batch_size=config['train_network']['batch_size'],
            complex_split=config['complex_split'])
    val_data   = np_to_tfdataset(
            X_valid, y_valid,
            batch_size=config['train_network']['batch_size'],
            complex_split=config['complex_split'])
    
    logging.info('')
    logging.info('Creating model...')
    train_params = config['train_network']
    strategy = tf.distribute.MirroredStrategy(devices=config['gpus'])
    with strategy.scope():
        model = get_model(
                model_type=config['model_type'], 
                loss_function=config['loss_function'],
                input_shape=config['input_shape'],
                load_model_path=train_params['load_model_path'],
                learning_rate=train_params['learning_rate'],
                window_size=train_params['window_size'],
                init_alpha=train_params['init_alpha'],
                noise_window_size=train_params['noise_window_size'])

    logging.info(model.summary())

    cb_list = get_training_cb(
            config['config_name'],
            patience=train_params['patience'],
            model_type=config['model_type'])

    history = model.fit(
            train_data,
            validation_data=val_data,
            batch_size=train_params['batch_size'],
            epochs=train_params['n_epochs'],
            initial_epoch=train_params['init_epoch'],
            callbacks=cb_list,
            shuffle=True)

    logging.info(history.history)
    logging.info('Training complete for config: {}'.format(config['config_name']))


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=argparse.FileType('r'))

    return parser

if __name__ == '__main__':
    main()

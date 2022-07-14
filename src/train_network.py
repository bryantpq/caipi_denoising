import argparse
import logging
import numpy as np
import os
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
    create_logger(config['config_name'])

    logging.info(config)
    logging.info('')
    logging.info('Loading training dataset from: {}'.format(config['data_folder']))
    X, y = load_dataset(config['data_folder'])

    shuffle_i = np.random.RandomState(seed=42).permutation(len(X))
    X, y = X[shuffle_i], y[shuffle_i]
    
    val_i = int( len(X) * config['train_network']['valid_split'] )
    X_train, y_train = X[:val_i], y[:val_i]
    X_valid, y_valid = X[val_i:], y[val_i:]
    
    del X, y
    
    logging.info('Train/Valid split:')
    logging.info('{}, {}, {}, {}'.format(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape))
    
    train_data = np_to_tfdataset(X_train, y_train)
    val_data   = np_to_tfdataset(X_valid, y_valid)
    
    logging.info('')
    logging.info('Creating model...')
    train_params = config['train_network']
    strategy = tf.distribute.MirroredStrategy(devices=config['gpus_to_use'])
    with strategy.scope():
        model = get_model(model_type=train_params['model_type'], 
                          input_shape=train_params['input_shape'],
                          load_model_path=train_params['load_model_path'],
                          learning_rate=train_params['learning_rate'])

    logging.info(model.summary())

    cb_list = get_training_cb(patience=train_params['patience'],
                              model_type=train_params['model_type'],
                              save_path=os.path.join(config['save_model_folder'], config['config_name']))
    
    history = model.fit(train_data,
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

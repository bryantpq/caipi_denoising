import argparse
import logging
import numpy as np
import os
import pdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/quahb/.conda/pkgs/cuda-nvcc-12.1.105-0'
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import yaml

from modeling.callbacks import get_training_cb
from modeling.get_model import get_model
from preparation.data_io import load_dataset
from preparation.prepare_tf_dataset import np_to_tfdataset, complex_split
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
    logging.info('Loading training dataset from: {}'.format(config['input_folder']))

    dimensions = config['dimensions']
    data_format = config['data_format']

    images_path = os.path.join(config['input_folder'], 'images')
    labels_path = os.path.join(config['input_folder'], 'labels')
    images = load_dataset(images_path, dimensions, data_format)
    labels = load_dataset(labels_path, dimensions, data_format)

    logging.info(f'Images, Labels dimensions: {images.shape}, {labels.shape}')

    #N_TO_USE = 3000 # should be at least 20
    #images, labels = images[:N_TO_USE], labels[:N_TO_USE]

    memory_used = images.nbytes * 2 / (1024 * 1024 * 1024)
    logging.info(f'Dataset size: {images.nbytes}  *  2  =  {memory_used} GB')

    if config['complex_split']:
        logging.info('Splitting complex data to real and imag channels')
        images = complex_split(images)
        labels = complex_split(labels)

    logging.debug(images.shape)
    logging.debug(labels.shape)

    shuffle_i = np.random.RandomState(seed=42).permutation(len(images))
    images, labels = images[shuffle_i], labels[shuffle_i]
    
    val_i = int( len(images) * config['valid_split'] )
    X_train, y_train = images[:val_i], labels[:val_i]
    X_valid, y_valid = images[val_i:], labels[val_i:]

    if config['trim_batch_end']:
        train_lim = X_train.shape[0] // config['batch_size'] * config['batch_size']
        valid_lim = X_valid.shape[0] // config['batch_size'] * config['batch_size']
        X_train, y_train = X_train[:train_lim], y_train[:train_lim]
        X_valid, y_valid = X_valid[:valid_lim], y_valid[:valid_lim]

    del images, labels
    
    network_params = config['network']
    strategy = tf.distribute.MirroredStrategy(devices=config['gpus'])
    with strategy.scope():
        train_data = np_to_tfdataset(X_train, y_train, batch_size=config['batch_size'])
        val_data   = np_to_tfdataset(X_valid, y_valid, batch_size=config['batch_size'])
        logging.info('Creating model...')
        model = get_model(dimensions, **network_params)

    logging.info(model.summary())
    logging.info(f'{strategy.num_replicas_in_sync}')
    logging.info('Train/Valid split:')
    logging.info('{}, {}, {}, {}'.format(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape))

    logging.info('')

    cb_list = get_training_cb(
            config_name,
            patience=config['patience'],
            model_type=network_params['model_type'])

    history = model.fit(
            train_data,
            validation_data=val_data,
            batch_size=config['batch_size'],
            epochs=config['n_epochs'],
            initial_epoch=config['init_epoch'],
            callbacks=cb_list,
            shuffle=True)

    logging.info(history.history)
    logging.info('Training complete for config: {}'.format(config_name))

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=argparse.FileType('r'))

    return parser

if __name__ == '__main__':
    main()

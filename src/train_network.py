import argparse
import logging
import numpy as np
import os
import socket
import pdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

hostname = socket.gethostname()
if 'titan' in hostname:
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/quahb/.conda/pkgs/cuda-nvcc-12.1.105-0'
elif 'compbio' in hostname or 'hpc' in hostname:
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/quahb/.conda/pkgs/cuda-nvcc-11.8.89-0'
else:
    raise ValueError(f'Unknown hostname: {hostname}')

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' # dont use the entire gpu(s)
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
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
    if args.fold is not None: config_name = config_name + f'_fold{args.fold}'

    create_logger(config_name, config['logging_level'])

    logging.info(config)
    logging.info('')

    dimensions = config['dimensions']
    data_format = config['data_format']

    # start: data loading
    logging.info('Loading training dataset from: {}'.format(config['input_folder']))
    images_path = os.path.join(config['input_folder'], 'images')
    labels_path = os.path.join(config['input_folder'], 'labels')

    if args.fold is not None: # load train/val following predefined train/val split of subjects
        FOLD_FILE = f'/home/quahb/caipi_denoising/data/five_fold_split/fold{args.fold}.yaml'
        with open(FOLD_FILE, 'r') as f:
            fold_split = yaml.safe_load(f)

        logging.info(f'Loading fold-{args.fold} subjects into train/val sets.')
        X_train = load_dataset(images_path, dimensions, data_format, ids=fold_split['train'])
        y_train = load_dataset(labels_path, dimensions, data_format, ids=fold_split['train'])
        X_valid = load_dataset(images_path, dimensions, data_format, ids=fold_split['valid'])
        y_valid = load_dataset(labels_path, dimensions, data_format, ids=fold_split['valid'])

        train_shuffle = np.random.RandomState(seed=42).permutation(len(X_train))
        valid_shuffle = np.random.RandomState(seed=42).permutation(len(X_valid))
        X_train, y_train = X_train[train_shuffle], y_train[train_shuffle]
        X_valid, y_valid = X_valid[valid_shuffle], y_train[valid_shuffle]

    else: # load entire dataset and then split into train/val
        images = load_dataset(images_path, dimensions, data_format)
        labels = load_dataset(labels_path, dimensions, data_format)

        logging.info(f'Images, Labels dimensions: {images.shape}, {labels.shape}')

        shuffle_i = np.random.RandomState(seed=42).permutation(len(images))
        images, labels = images[shuffle_i], labels[shuffle_i]

        TRAIN_SIZE = 0.8
        logging.info(f"Splitting whole dataset into train/validation: {TRAIN_SIZE}/{1 - TRAIN_SIZE}")
        val_i = int( len(images) * TRAIN_SIZE )
        X_train, y_train = images[:val_i], labels[:val_i]
        X_valid, y_valid = images[val_i:], labels[val_i:]

        del images, labels

    if 'compleximage_2d_full' in config_name:
        logging.info('taking a subset of the full training data or this shit dont work')
        X_train = X_train[:7000]
        y_train = y_train[:7000]

    logging.info('Train/Valid split:')
    logging.info('{}, {}, {}, {}'.format(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape))
    
    # start: network loading
    network_params = config['network']
    logging.debug('Available GPUs:')
    for i in tf.config.list_physical_devices('GPU'): logging.debug(f'    {i}')

    batch_size, trim_batch = config['batch_size'], config['trim_batch_end']
    strategy = tf.distribute.MirroredStrategy(devices=config['gpus'])
    with strategy.scope():
        train_data = np_to_tfdataset(X_train, y_train, batch_size=batch_size, trim_batch=trim_batch)
        val_data   = np_to_tfdataset(X_valid, y_valid, batch_size=batch_size, trim_batch=trim_batch)
        logging.info('Creating model...')
        model = get_model(dimensions, **network_params)

    model.summary(print_fn=logging.debug)

    cb_list = get_training_cb(
            config_name,
            patience=config['patience'],
            model_type=network_params['model_type']
    )

    history = model.fit(
            train_data,
            validation_data=val_data,
            batch_size=config['batch_size'],
            epochs=config['n_epochs'],
            initial_epoch=config['init_epoch'],
            callbacks=cb_list,
            shuffle=True
    )

    logging.info(history.history)
    logging.info(f'Epochs Trained: {len(history.history)}, Patience: {config["patience"]}')
    logging.info(f'Training complete for config: {config_name}')

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=argparse.FileType('r'))
    parser.add_argument('--fold', type=int, choices=[1,2,3,4,5])

    return parser

if __name__ == '__main__':
    main()

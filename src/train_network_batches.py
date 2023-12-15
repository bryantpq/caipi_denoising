import argparse
import datetime
import logging
import multiprocessing
import numpy as np
import os
import socket
import pdb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

hostname = socket.gethostname()
if 'titan' in hostname:
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/quahb/.conda/pkgs/cuda-nvcc-12.1.105-0'
elif hostname in ['compbio', 'hpc']:
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
    if args.fold is not None:
        config_name = config_name + f'_fold{args.fold}'
    else:
        raise NotImplementedError('train_network_batches.py requires fold number')

    create_logger(config_name, config['logging_level'])

    logging.info(config)
    logging.info('')

    dimensions = config['dimensions']
    data_format = config['data_format']
    epoch, batch = args.epoch, args.batch

    if batch == -1:
        logging.info('Validating epoch {epoch} on validation set')
    else:
        logging.info(f'Training epoch {epoch} on batch {batch}')

    # start: data loading
    logging.info('Loading training dataset from: {}'.format(config['input_folder']))
    images_path = os.path.join(config['input_folder'], 'images')
    labels_path = os.path.join(config['input_folder'], 'labels')

    FOLD_FILE = f'/home/quahb/caipi_denoising/data/five_fold_split/fold{args.fold}.yaml'
    with open(FOLD_FILE, 'r') as f:
        fold_split = yaml.safe_load(f)
    
    SIZE = 4
    start, stop = batch * SIZE - SIZE, batch * SIZE # [0, 4], [4, 8], [8, 12] ...

    logging.info(f'Loading fold-{args.fold} subjects.')
    if batch == -1: # validation batch
        images = load_dataset(images_path, dimensions, data_format, ids=fold_split['valid'])
        labels = load_dataset(labels_path, dimensions, data_format, ids=fold_split['valid'])
    elif batch == 11: # last training batch, variable number of subjects to laod
        last_n = len(fold_split['train']) % SIZE
        start = batch * SIZE - SIZE
        stop  = batch * SIZE - SIZE + last_n # [40, 4X]
        images = load_dataset(images_path, dimensions, data_format, ids=fold_split['train'], batch=[start, stop])
        labels = load_dataset(labels_path, dimensions, data_format, ids=fold_split['train'], batch=[start, stop])
    else:
        start, stop = batch * SIZE - SIZE, batch * SIZE # [0, 4], [4, 8], ..., [36, 40]
        images = load_dataset(images_path, dimensions, data_format, ids=fold_split['train'], batch=[start, stop])
        labels = load_dataset(labels_path, dimensions, data_format, ids=fold_split['train'], batch=[start, stop])

    logging.info(f'Images, Labels dimensions: {images.shape}, {labels.shape}')

    shuffle_i = np.random.RandomState(seed=42).permutation(len(images))
    images, labels = images[shuffle_i], labels[shuffle_i]
    
    # start: network loading
    network_params = config['network']
    model_folder = f"/home/quahb/caipi_denoising/models/{config_name}/"
    model_name   = f"{network_params['model_type']}" + "_ep{}_bat{}"
    if epoch == 1 and batch == 1:
        network_params['load_model_path'] = None
    elif epoch != 1 and batch == 1:
        network_params['load_model_path'] = os.path.join(model_folder, model_name.format(epoch - 1, 11))
    elif batch > 1:
        network_params['load_model_path'] = os.path.join(model_folder, model_name.format(epoch,batch-1))
    elif batch == -1: # run validation on final trained model of the epoch
        network_params['load_model_path'] = os.path.join(model_folder, model_name.format(epoch, 11))
    else:
        raise ValueError('Oops')

    logging.debug('Available GPUs:')
    for i in tf.config.list_physical_devices('GPU'): logging.debug(f'    {i}')

    batch_size, trim_batch = config['batch_size'], config['trim_batch_end']
    if not trim_batch:
        logging.info(f'Overriding trim_batch value: {trim_batch}\nSetting it to True')
        trim_batch = True

    strategy = tf.distribute.MirroredStrategy(devices=config['gpus'])
    with strategy.scope():
        data = np_to_tfdataset(images, labels, batch_size=batch_size, trim_batch=trim_batch)
        logging.info('Creating model...')
        model = get_model(dimensions, **network_params)

    if batch != -1: # train model on training set
        history = model.fit(
                data,
                batch_size=config['batch_size'],
                epochs=epoch,
                initial_epoch=epoch - 1,
                shuffle=True
        )
        loss = history.history['loss']
        logging.info(loss)
        log_loss(loss, epoch, batch)

        save_name = os.path.join(model_folder, model_name.format(epoch, batch))
        model.save_weights(save_name)
        logging.info(f'Saving model:    {save_name}')
    else: # run model performance on validation set
        loss = model.evaluate(
                data, batch_size=config['batch_size']
        )
        logging.info(loss)
        log_loss(loss, epoch, batch)

def log_loss(loss, epoch, batch):
    today = str(datetime.date.today())
    fname = f'../logs/{config_name}_{today}.txt'
    with open(fname, 'a') as f:
        now = str(datetime.datetime.now())
        if batch != -1:
            f.write(f'{now}:Epoch {epoch}:Batch {batch}:Train Loss {loss}')
        else:
            f.write(f'{now}:Epoch {epoch}:Valid Loss {loss}')

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=argparse.FileType('r'))
    parser.add_argument('--fold', type=int, choices=[1,2,3,4,5])
    parser.add_argument('epoch', type=int)
    parser.add_argument('batch', type=int)

    return parser

if __name__ == '__main__':
    main()

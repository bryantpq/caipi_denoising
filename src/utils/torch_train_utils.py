import logging
import math
import numpy as np
import os
import torch
import yaml

from datetime import datetime, date
from preparation.data_io import load_dataset
from preparation.prepare_dataset import merge_datasets
from sigfig import round
from torch.utils.tensorboard import SummaryWriter

FOLD_FILE = '/home/quahb/caipi_denoising/data/five_fold_split/fold{}.yaml'
MERGE_TRAIN_VAL, TRAIN_SIZE = False, 0.8

def get_data_gen(
        fold, images_path, labels_path, dimensions, data_format, 
        dataset_type=None,
        rank=None,
        subject_batch_size=None,
):
    assert dataset_type in ['train', 'valid']

    if fold is not None:
        with open(FOLD_FILE.format(fold), 'r') as f: fold_dict = yaml.safe_load(f)
        n_subjects = len(fold_dict[dataset_type])
    else:
        n_subjects = len([ p for p in os.listdir(images_path) if p.split('.')[-1] == 'npy' ])

    if type(subject_batch_size) == int and subject_batch_size > 0:
        n_full_batches = n_subjects // subject_batch_size
        batch_indices = []
        for batch_i in range(1, n_full_batches):
            start = batch_i * subject_batch_size - subject_batch_size
            stop  = batch_i * subject_batch_size
            batch_indices.append( [start, stop] )

        last_n = n_subjects % subject_batch_size
        if last_n != 0:
            start = n_full_batches * subject_batch_size
            stop  = n_full_batches * subject_batch_size + last_n
            batch_indices.append( [start, stop] )
    else:
        batch_indices = [ None ]

    for i, batch_i in enumerate(batch_indices):
        if fold is not None:
            logging.info(f'Rank {rank}: Loading {dataset_type} set from fold-{fold} batch {i+1}/{len(batch_indices)}.')
            X, y = load_dataset(
                    [images_path, labels_path], 
                    dimensions, 
                    data_format, 
                    ids=fold_dict[dataset_type], 
                    batch=batch_i, 
                    rank=rank
            )

            shuffle = np.random.RandomState(seed=42).permutation(len(X))
            X, y = X[shuffle], y[shuffle]

        else: # load entire dataset and then split into train/val
            images, labels = load_dataset(
                    [images_path, labels_path],
                    dimensions,
                    data_format,
                    batch=batch_i,
                    rank=rank
            )
            logging.info(f'Loading batch {i+1}/{len(batch_indices)} into {dataset_type} set.')
            logging.info(f'Images, Labels dimensions: {images.shape}, {labels.shape}')

            shuffle = np.random.RandomState(seed=42).permutation(len(images))
            images, labels = images[shuffle], labels[shuffle]

            logging.info(f"Splitting whole dataset into train/validation: {TRAIN_SIZE}/{1 - TRAIN_SIZE}")
            val_i = int( len(images) * TRAIN_SIZE )
            if dataset_type == 'train':
                X, y = images[:val_i], labels[:val_i]
            elif dataset_type == 'valid':
                X, y = images[val_i:], labels[val_i:]

            del images, labels

        # (N, H, W, C) --> (N, C, H, W)
        X, y = map( lambda x: np.moveaxis(x, -1, 1), (X, y) )
        X, y = map( torch.tensor, (X, y) )

        yield X, y

def train_one_epoch(model, loss_fn, optimizer, train_loader, epoch_index, tb_writer, rank):
    last_loss, running_loss = 0., 0.
    n_batches = len(train_loader)

    for i, data in enumerate(train_loader):
        images, labels = data
        loss = batch_loss(model, images, labels, loss_fn, rank)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % n_batches == n_batches - 1 and True:
            last_loss = running_loss / n_batches
            logging.info('    Rank {} Batch {}-th loss: {}'.format(rank, i + 1, round(last_loss, sigfigs=5)))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def setup_paths(config_name):
    TB_PATH = '/home/quahb/caipi_denoising/logs/tensorboard/{}_{}/'
    writer = SummaryWriter(TB_PATH.format(config_name, date.today()))

    save_path = os.path.join('/home/quahb/caipi_denoising/models/', 
            '{}_{}'.format(config_name, str(date.today())))

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        logging.info(f"Creating new folder for model weights: {save_path}")

    return writer, save_path

def batch_loss(model, images, labels, loss_fn, rank):
    images, labels = images.to(rank), labels.to(rank)
    
    outputs = model(images)
    outputs = outputs.to(rank)

    loss = loss_fn(outputs, labels)

    return loss

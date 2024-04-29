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
TRAIN_SIZE, VAL_SIZE = 1.0, 0.0

def batch_loss(model, images, labels, loss_fn, rank):
    images, labels = images.to(rank), labels.to(rank)
    
    outputs = model(images)
    outputs = outputs.to(rank)

    loss = loss_fn(outputs, labels)

    return loss

def setup_paths(config_name, load_train_state):
    if load_train_state is None:
        out_folder = '{}_{}'.format(config_name, date.today())
    else:
        loaded_folder = load_train_state.split('/')[-2]
        out_folder = '{}'.format(loaded_folder)

    writer = SummaryWriter('/home/quahb/caipi_denoising/logs/tensorboard/{}/'.format(out_folder))
    save_path = os.path.join('/home/quahb/caipi_denoising/models/{}/'.format(out_folder))

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        logging.info(f"Creating new folder for model weights: {save_path}")

    return writer, save_path

def train_one_epoch(model, loss_fn, optimizer, train_loader, epoch, tb_batch_id, tb_writer, rank, world_size=4):
    last_loss, running_loss, n_batches = 0., 0., len(train_loader)

    for i, data in enumerate(train_loader):
        images, labels = data
        loss = batch_loss(model, images, labels, loss_fn, rank)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    last_loss = running_loss / n_batches
    logging.info('  Rank {} Batch ID {} loss: {}'.format(
        rank, tb_batch_id, round(last_loss, sigfigs=5)
    ))

    last_loss = torch.tensor([last_loss]).to(rank)
    batch_losses = [torch.zeros(1, dtype=torch.float32).to(rank) for _ in range(world_size)]
    torch.distributed.all_gather(batch_losses, last_loss)

    avg_batch_loss = torch.mean(torch.stack(batch_losses))
    tb_writer.add_scalar('Loss/Batch loss', avg_batch_loss, tb_batch_id)
    tb_writer.flush()

    return avg_batch_loss

def get_data_gen(
        fold, images_path, labels_path, dimensions, data_format, 
        dataset_type=None,
        rank=None,
        subject_batch_size=None,
):
    if fold is not None:
        with open(FOLD_FILE.format(fold), 'r') as f: fold_dict = yaml.safe_load(f)
        n_subjects = len(fold_dict[dataset_type])
    else:
        n_subjects = len([ p for p in os.listdir(images_path) if p.split('.')[-1] == 'npy' ])

    if type(subject_batch_size) == int and subject_batch_size > 0:
        n_full_batches = n_subjects // subject_batch_size
        batch_indices = []
        for batch_i in range(n_full_batches):
            start = batch_i * subject_batch_size
            stop  = batch_i * subject_batch_size + subject_batch_size
            batch_indices.append( [start, stop] )

        last_n = n_subjects % subject_batch_size
        if last_n != 0:
            start = n_full_batches * subject_batch_size
            stop  = n_full_batches * subject_batch_size + last_n
            batch_indices.append( [start, stop] )
    else:
        batch_indices = [ None ]

    assert batch_indices[-1][-1] == n_subjects
    if rank == 0: logging.info(f'Splitting {n_subjects}-subjects into {len(batch_indices)}-batches...')

    for i, batch_i in enumerate(batch_indices):
        if fold is not None:
            if rank is not None: logging.info(f'Rank {rank}: Loading {dataset_type} set from fold-{fold} batch {i+1}/{len(batch_indices)}.')
            else: logging.info(f'Loading {dataset_type} set from fold-{fold} batch {i+1}/{len(batch_indices)}.')

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
            if rank is not None: logging.info(f'Rank {rank}: Loading {dataset_type} set from fold-{fold} batch {i+1}/{len(batch_indices)}.')
            else: logging.info(f'Loading {dataset_type} set from fold-{fold} batch {i+1}/{len(batch_indices)}.')

            images, labels = load_dataset(
                    [images_path, labels_path],
                    dimensions,
                    data_format,
                    batch=batch_i,
                    rank=rank
            )
            logging.info(f'Images, Labels dimensions: {images.shape}, {labels.shape}')

            shuffle = np.random.RandomState(seed=42).permutation(len(images))
            images, labels = images[shuffle], labels[shuffle]

            val_i = int( len(images) * TRAIN_SIZE )
            if dataset_type == 'train':
                logging.info(f"Splitting whole dataset into train/validation: {TRAIN_SIZE}/{1 - TRAIN_SIZE}")
                X, y = images[:val_i], labels[:val_i]
            elif dataset_type == 'valid':
                logging.info(f"Splitting whole dataset into train/validation: {TRAIN_SIZE}/{1 - TRAIN_SIZE}")
                X, y = images[val_i:], labels[val_i:]
            elif dataset_type == 'train_valid':
                X, y = images, labels

            del images, labels

        # (N, H, W, C) --> (N, C, H, W)
        X, y = map( lambda x: np.moveaxis(x, -1, 1), (X, y) )
        X, y = map( torch.tensor, (X, y) )

        yield X, y

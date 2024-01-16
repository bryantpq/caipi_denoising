import argparse
import logging
import numpy as np
import os
import socket
import torch
import torch.multiprocessing as mp
import pdb
import yaml

from datetime import datetime, date
from sigfig import round
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from modeling.torch_complex_utils import complex_mse
from modeling.torch_models import get_model
from preparation.data_io import load_dataset
from preparation.preprocessing_pipeline import rescale_magnitude
from utils.create_logger import create_logger
from utils.torch_train_utils import batch_loss, get_data_gen, train_one_epoch, setup_paths


RUN_VALIDATION = False

def main(rank, world_size):
    ddp_setup(rank, world_size)
    parser = create_parser()
    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    config_file = args.config.name.split('/')[-1]
    config_name = config_file.split('.')[0]
    if args.fold is not None: config_name = config_name + f'_fold{args.fold}'
    if args.subjects is not None: subject_batch_size = args.subjects

    create_logger(config_name, config['logging_level'])

    if rank == 0:
        logging.info(config)
        logging.info('')

    batch_size = config['batch_size']
    data_format = config['data_format']
    dimensions = config['dimensions']
    network_params = config['network']

    writer, save_path = setup_paths(config_name)
    model_save_name = os.path.join(save_path, network_params['model_type'] + '_ep{}.pt')

    # start: data loading
    if rank == 0: logging.info('Loading training dataset from: {}'.format(config['input_folder']))
    images_path = os.path.join(config['input_folder'], 'images')
    labels_path = os.path.join(config['input_folder'], 'labels')

    model = get_model(network_params['model_type'], dimensions).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    best_vloss = 1000000.
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=network_params['learning_rate'])

    for epoch in range(config['n_epochs']):
        if rank == 0: logging.info('********    Epoch {}    ********'.format(epoch + 1))

        train_gen = get_data_gen(
                args.fold, images_path, labels_path, dimensions, data_format, 
                dataset_type='train',
                rank=rank,
                subject_batch_size=subject_batch_size,
        )

        model.train(True)
        for batch_i, (X_train, y_train) in enumerate(train_gen):
            if rank == 0: logging.info(f'Subject Batch {batch_i} -- X_train, y_train: {X_train.shape}, {y_train.shape}')

            train_set = TensorDataset(X_train, y_train)

            train_sampler = DistributedSampler(
                    train_set, 
                    num_replicas=world_size, 
                    rank=rank, 
                    shuffle=False, 
                    drop_last=True
            )

            train_loader = DataLoader(
                    train_set, 
                    batch_size=batch_size, 
                    pin_memory=True,
                    shuffle=False, 
                    sampler=train_sampler
            )

            if rank == 0: logging.info('Batch size: {}, Num batches: Train: {}'.format(batch_size, len(train_loader)))

            avg_loss = train_one_epoch(model, loss_fn, optimizer, train_loader, epoch, writer, rank)
            del train_set, train_sampler, train_loader

            if rank == 0: 
                logging.info('')
                logging.info(f'Epoch {epoch}: Completed Subject Batch {batch_i} ...')

        if rank == 0: # calculate vloss and save model
            save_name = model_save_name.format(epoch + 1)
            logging.info(f'    Saving model {save_name}')
            torch.save(model.module.state_dict(), save_name)

            if RUN_VALIDATION:
                model.eval()
                valid_gen = get_data_gen(
                        args.fold, images_path, labels_path, dimensions, data_format,
                        dataset_type='valid',
                        rank=rank,
                        subject_batch_size=8,
                )

                X_valid, y_valid = list(valid_gen)[0]

                valid_set = TensorDataset(X_valid, y_valid)
                valid_loader = DataLoader(
                        valid_set,
                        batch_size=batch_size, 
                        shuffle=False, 
                )
                logging.info('Batch size: {}, Num batches: Valid: {}'.format(batch_size, len(valid_loader)))

                running_vloss = 0.0
                with torch.no_grad():
                    for i, vdata in enumerate(valid_loader):
                        vimages, vlabels = vdata
                        vloss = batch_loss(model, vimages, vlabels, loss_fn, rank)
                        running_vloss += vloss

                avg_vloss = running_vloss / (i + 1)
                logging.info('    Average Loss: Train {}, Valid {}'.format(
                        round(avg_loss, sigfigs=5), round(avg_vloss.item(), sigfigs=5)))

                writer.add_scalars('Training vs. Validation Loss',
                        { 'Training': avg_loss, 'Validation': avg_vloss }, epoch + 1)
                writer.flush()

                if avg_vloss < best_vloss:
                    best_vloss = avg_vloss.item()
                    logging.info('    Valid loss {} better than {}'.format(
                            round(avg_vloss.item(), sigfigs=5), round(best_vloss, sigfigs=5)))

    destroy_process_group()
    logging.info(f'Training complete for config: {config_name}')

def ddp_setup(rank, world_size):
    '''
    rank: unique identifier of each process
    world_size: total number of processes
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=argparse.FileType('r'))
    parser.add_argument('--fold', type=int, choices=[1,2,3,4,5,0])
    parser.add_argument('--subjects', type=int, default=4)

    return parser

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)

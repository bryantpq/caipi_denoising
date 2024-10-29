import argparse
import datetime
import logging
import numpy as np
import os
import socket
import torch
import torch.multiprocessing as mp
import pdb
import yaml

from sigfig import round
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from modeling.torch_complex_utils import complex_mse
from modeling.torch_models import get_model, get_loss
from preparation.preprocessing_pipeline import rescale_magnitude
from utils.create_logger import create_logger
from utils.train_torch_utils import batch_loss, get_data_gen, train_one_epoch, setup_paths

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

RUN_VALIDATION = True

def main(rank, world_size):
    ddp_setup(rank, world_size)
    parser = create_parser()
    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    config_file = args.config.name.split('/')[-1]
    config_name = config_file.split('.')[0]
    if args.fold is not None: config_name = config_name + f'_fold{args.fold}'
    if args.subjects is not None: subject_batch_size = args.subjects

    create_logger(config_name, args.logging)

    if rank == 0:
        logging.info(config)
        logging.info('')

    batch_size  = config['batch_size']
    data_format = config['data_format']
    dimensions  = config['dimensions']
    n_epochs    = config['n_epochs']
    network     = config['network']
    model_type  = network['model_type']

    images_path = os.path.join(config['input_folder'], 'images')
    labels_path = os.path.join(config['input_folder'], 'labels')

    if 'run_validation' in config: run_validation = config['run_validation']
    tb_writer, save_path = setup_paths(config_name, config['load_train_state'])

    # start: data loading
    if rank == 0: logging.info(f"Loading {args.dataset_type} dataset from: {config['input_folder']}")

    model = get_model(
            model_type, 
            dimensions, 
            n_hidden_layers=network['n_hidden_layers'],
            residual_layer=network['residual_layer'],
            load_model_path=network['load_model_path']
    )
    if rank == 0: logging.debug(model)
    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    loss_fn = get_loss(config_name, network['loss'])

    train_start_time = datetime.datetime.now()
    optimizer = torch.optim.Adam(model.parameters(), lr=network['learning_rate'])
    if network['decay_lr'] is not None: scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # lr at epoch x : lr(x) = init_lr * gamma**x
    # lr = lambda x: init_lr * gamma**x

    if config['load_train_state'] is not None:
        logging.info(f'Loading previous train state: {config["load_train_state"]}')
        checkpoint = torch.load(config['load_train_state'])

        init_epoch  = checkpoint['epoch']
        tb_batch_id = checkpoint['tb_batch_id']
        optimizer.load_state_dict(checkpoint['optimizer'])
        if network['decay_lr'] is not None: scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        init_epoch = 0
        tb_batch_id = 0

    best_vloss = 100000.0
    best_epoch_vloss = -1
    for epoch in range(init_epoch, n_epochs):
        if rank == 0:
            logging.info('******************            Epoch {}/{}            ******************'.format(epoch + 1, n_epochs))
            if network['decay_lr'] is not None:
                lr = scheduler.get_last_lr()[0]
            else:
                lr = network['learning_rate']
            logging.info('Learning rate: {}'.format(lr))
            tb_writer.add_scalar('Learning Rate', lr, epoch + 1)
            tb_writer.flush()
        subj_batch_losses = []
        epoch_start_time = datetime.datetime.now()

        train_gen = get_data_gen(
                args.fold, images_path, labels_path, dimensions, data_format, 
                dataset_type=args.dataset_type,
                rank=rank,
                subject_batch_size=subject_batch_size,
        )

        model.train(True)
        for batch_i, (X_train, y_train) in enumerate(train_gen):
            tb_batch_id += 1
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

            if rank == 0: logging.info(f'Subject Batch {batch_i + 1}; Batch size: {batch_size}; Num Batches: {len(train_loader)}; X_train, y_train: {X_train.shape}, {y_train.shape}')

            avg_loss = train_one_epoch(
                    model, 
                    loss_fn, 
                    optimizer, 
                    train_loader,
                    epoch,
                    tb_batch_id,
                    tb_writer,
                    rank,
                    world_size
            )
            subj_batch_losses.append(avg_loss)

            del train_set, train_sampler, train_loader

            if rank == 0:
                logging.debug(f'Epoch {epoch + 1}: Completed Subject Batch {batch_i + 1}')

            # close for-loop training data

        if network['decay_lr'] is not None: scheduler.step()
        epoch_end_time = datetime.datetime.now()
        epoch_elapsed_sec = epoch_end_time - epoch_start_time

        epoch_loss = torch.mean(torch.stack(subj_batch_losses))
        # log to tensorboard, save model, calculate vloss
        if rank == 0: 
            tb_writer.add_scalar('Loss/Epoch', epoch_loss, epoch + 1)
            tb_writer.flush()
            logging.info(f'Completed fold-{args.fold} Epoch {epoch + 1}/{n_epochs}: {epoch_elapsed_sec}, Loss: {epoch_loss}')

            model_save_name = os.path.join(save_path, model_type + '_ep{}.pt')
            model_save_name = model_save_name.format(epoch + 1)
            torch.save(model.module.state_dict(), model_save_name)

            state_save_name = os.path.join(save_path, 'ckpt_ep{}.pth'.format(epoch + 1))
            train_state = {
                    'epoch': epoch + 1,
                    'tb_batch_id': tb_batch_id,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if network['decay_lr'] is not None else None,
            }
            torch.save(train_state, state_save_name)
            logging.info(f'    Saving model {model_save_name}')
            logging.info(f'    Saving train state {state_save_name}')

            #if epoch % 10 == 9: # TODO save feature maps as images
            #    if model_type == 'dcsrn':
            #        tb_writer.add_images(f'Epoch {epoch} final layer', epoch)

        if RUN_VALIDATION:
            if rank == 0: logging.info('Running validation...')
            model.eval()

            VAL_SUBJECTS = 5
            valid_gen = get_data_gen(
                    args.fold, images_path, labels_path, dimensions, data_format,
                    dataset_type='valid',
                    rank=rank,
                    subject_batch_size=VAL_SUBJECTS,
            )

            running_vloss = 0.0
            for batch_i, (X_valid, y_valid) in enumerate(valid_gen):
                valid_set = TensorDataset(X_valid, y_valid)
                valid_loader = DataLoader(
                        valid_set,
                        batch_size=batch_size, 
                        shuffle=False, 
                )
                if rank == 0: logging.info('Valid Batch size: {}, Num batches: {}; X_valid, y_valid: {}, {}'.format(batch_size, len(valid_loader), X_valid.shape, y_valid.shape))

                with torch.no_grad():
                    for i, vdata in enumerate(valid_loader):
                        vimages, vlabels = vdata
                        vloss = batch_loss(model, vimages, vlabels, loss_fn, rank)
                        running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            train_loss, valid_loss = round(epoch_loss.item(), sigfigs=5), round(avg_vloss.item(), sigfigs=5)
            if rank == 0: logging.info('    Epoch {} Average Loss: Train {}, Valid {}'.format(epoch + 1, train_loss, valid_loss))

            tb_writer.add_scalars(
                    'Loss/Train vs Valid',
                    {
                        'Train': epoch_loss,
                        'Valid': avg_vloss,
                    },
                    epoch + 1
            )
            tb_writer.flush()

            if avg_vloss < best_vloss:
                if rank == 0: logging.info('    Valid loss {} better than {}'.format(valid_loss, best_vloss))
                best_vloss = avg_vloss.item()
                best_epoch_vloss = epoch + 1
            if rank == 0 and epoch < n_epochs:
                if RUN_VALIDATION: logging.info(f'    Best validation at epoch: {best_epoch_vloss} with {best_vloss}')
            # close validation block
        # close epoch for loop

    total_train_time = epoch_end_time - train_start_time
    tb_writer.close()
    destroy_process_group()

    if rank == 0:
        if RUN_VALIDATION: logging.info(f'Best validation at epoch: {best_epoch_vloss} with {best_vloss}')
        logging.info(f'Time taken to train {epoch + 1} epochs: {total_train_time}')
        logging.info(f'Training complete for config: {config_name}')
        logging.info(config)

def ddp_setup(rank, world_size):
    '''
    rank: unique identifier of each process
    world_size: total number of processes
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size, timeout=datetime.timedelta(hours=1))
    torch.cuda.set_device(rank)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=argparse.FileType('r'))
    parser.add_argument('--fold', type=int, choices=[1,2,3,4,5,0])
    parser.add_argument('--dataset_type', default='train', choices=['train', 'overfit_one', 'train_valid', 'test'])
    parser.add_argument('--subjects', type=int, default=1)
    parser.add_argument('--logging', default='info', choices=['info', 'debug'])

    return parser

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)

import argparse
import logging
import nibabel as nib
import numpy as np
import os
import pdb
import torch
import torch.multiprocessing as mp
import yaml

from patchify import patchify, unpatchify
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from tqdm import tqdm

from modeling.torch_models import get_model
from preparation.data_io import write_data


def main(rank, world_size):
    ddp_setup(rank, world_size)
    parser = create_parser()
    args = parser.parse_args()

    if args.dimensions in [2, 3]: input_shape = [None] + args.input_size + [1]

    print(f'Model {args.model_type}: {args.model_path}')
    model = get_model(args.model_type, args.dimensions)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    if args.type_dir:
        files_to_load = sorted(os.listdir(args.input_path))
        files_to_load = [ os.path.join(args.input_path, f) for f in files_to_load if '.npy' in f or '.nii.gz' in f ]
        if args.fold is not None:
            FOLD_FILE = f'/home/quahb/caipi_denoising/data/five_fold_split/fold{args.fold}.yaml'
            with open(FOLD_FILE, 'r') as f: fold_split = yaml.safe_load(f)

            keep = []
            for f in files_to_load:
                subj_id = f.split('/')[-1] # /home/dir/file.npy
                subj_id = '_'.join(subj_id.split('_')[:-1]) # 1_02_030-V1_CAIPI1x3.npy
                if subj_id in fold_split['test']: keep.append(f)
            print(f'Running model prediction on subjects from fold-{args.fold}:')
            #print(keep)
            files_to_load = keep
        print(f'Found {len(files_to_load)} files at {args.input_path} to denoise.')
    else:
        files_to_load = [ args.input_path ]

    for cur_file in tqdm(files_to_load[:2], ncols=80):
        # load data
        print(f'\nLoading file: {cur_file}...')
        fname, fext = os.path.splitext(cur_file)
        if 'gz' in fext:
            data = np.array(nib.load(cur_file).dataobj)
        elif 'npy' in fext:
            data = np.load(cur_file)

        # prediction stage
        if args.extract_patches:
            patch_size, extract_step = args.input_size, args.extract_step
            if args.dimensions == 2:
                if len(patch_size) == 2:
                    patch_size = patch_size + [1]
                if len(extract_step) == 2:
                    extract_step = extract_step + [1]

            patches = patchify(data, patch_size, extract_step)
            before_patches = np.copy(patches)
            patches_shape = patches.shape
            patches = patches.reshape(-1, *patch_size)
            patches = np.expand_dims(patches, axis=-1) # 3D patches: (n_patches, X, Y, Z, 1)

            # convert to torch format
            patches = np.moveaxis(patches, -1, 1)
            patches = map(torch.tensor, patches)
            patches = TensorDataset(patches)
            sampler = DistributedSampler(patches, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
            patches_loader = DataLoader(patches_loader, batch_size=32, shuffle=False, sampler=sampler)

            # run prediction
            patches_out = []
            for patches_batch in patches_loader:
                patches_batch = patches_batch[0]
                patches_batch = patches_batch.to(rank)
                res = model(
                        patches_batch
                )
                res = res.cpu().detach().numpy()
                patches_out.append(res)
            patches_out = np.moveaxis(patches, 1, -1)
            patches = np.squeeze(patches_out)

            output = unpatchify(patches, data.shape)
        else:
            # data.shape 384, 384, 256
            data = np.moveaxis(data, -1, 0) # 256, 384, 384
            data = np.expand_dims(data, axis=3) # 256, 384, 384, 1

            # convert to torch format
            data = np.moveaxis(data, -1, 1) # 256, 1, 384, 384
            data = torch.tensor(data)
            data = TensorDataset(data)
            sampler = DistributedSampler(data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
            data_loader = DataLoader(data, batch_size=4, shuffle=False, sampler=sampler)

            # run prediction
            data_out = []
            for data_batch in data_loader:
                data_batch = data_batch[0]
                data_batch = data_batch.to(rank)
                res = model(
                        data_batch
                )
                #res = res.cpu().detach().numpy()
                #data_out.append(res)
            gather_res = [ None for _ in range(world_size) ]
            torch.distributed.gather(gather_res, res)
            pdb.set_trace()
            data = np.vstack(data_out)

            data = np.moveaxis(data, 1, -1)
            data = np.squeeze(data)
            output = np.moveaxis(data, 0, -1)

        if args.type_dir: 
            f_name = cur_file.split('/')[-1]
            out_name = os.path.join(args.output_path, f_name)
        else:
            out_name = args.output_path

        if np.iscomplexobj(output):
            save_dtype = 'complex64'
        else:
            save_dtype = 'float32'

        # gather tensors here
        if rank == 0:
            if args.extract_patches and args.debug_patches:
                before_patches_name = '/'.join(out_name.split('/')[:-1]) + '/before_patches/' + out_name.split('/')[-1]
                after_patches_name = '/'.join(out_name.split('/')[:-1]) + '/after_patches/' + out_name.split('/')[-1]
                write_data(before_patches, before_patches_name, save_dtype, save_format=fext)
                write_data(patches, after_patches_name, save_dtype, save_format=fext)
                print(f'Saving debug patches {after_patches_name}')

            print(f'Saving {out_name}')
            print('test')
            write_data(output, out_name, save_dtype, save_format=fext)

        torch.distributed.destroy_process_group()


    print('Complete!')

def ddp_setup(rank, world_size):
    '''
    rank: unique identifier of each process
    world_size: total number of processes
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def create_parser():
    parser = argparse.ArgumentParser()
    example_str = 'python predict.py 2 cdncnn mse --type_dir ../data/datasets/accelerated/msrebs_all_testing/{inputs,outputs} ../models/compleximage_full_cdncnn_2022-12-15/complex_dncnn_ep26.h5 384 384'
    parser.add_argument('dimensions', type=int, choices=[2, 3])
    parser.add_argument('model_type', choices=['dncnn', 'cdncnn'])
    parser.add_argument('loss_function', choices=['mse'])
    parser.add_argument('--type_dir', action='store_true', help='input_path/output_path will be intepreted as dirs')
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('model_path')
    parser.add_argument('input_size', nargs='+', type=int, help='[ length width ]')
    parser.add_argument('--extract_patches', action='store_true')
    parser.add_argument('--extract_step', nargs='+', type=int, help='[ length width ]')
    parser.add_argument('--fold', type=int, choices=[1,2,3,4,5,6])
    parser.add_argument('--debug_patches', action='store_true')

    return parser

if __name__ == '__main__':
    import sys
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)

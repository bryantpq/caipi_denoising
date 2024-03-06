import argparse
import datetime
import logging
import nibabel as nib
import numpy as np
import os
import pdb
import torch
import yaml

from patchify import patchify, unpatchify
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from modeling.torch_models import get_model
from preparation.data_io import write_data

FULL_BATCH_SIZE = 8
PATCH_BATCH_SIZE = 8 # 4 lowest
N_HIDDEN_LAYERS = 12
RESIDUAL_LAYER  = True

def main():
    parser = create_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = args.model_path.split('/')[-1].split('_')[0]

    print(f'Model {model_type}: {args.model_path}')
    model = get_model(model_type, args.dimensions, N_HIDDEN_LAYERS, RESIDUAL_LAYER)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to(device)

    if args.type_dir:
        files_to_load = sorted(os.listdir(args.input_path))
        files_to_load = [ os.path.join(args.input_path, f) for f in files_to_load if '.npy' in f or '.nii.gz' in f ]
        if args.fold is not None:
            FOLD_FILE = f'/home/quahb/caipi_denoising/data/five_fold_split/fold{args.fold}.yaml'
            with open(FOLD_FILE, 'r') as f: fold_split = yaml.safe_load(f)

            keep = []
            for f in files_to_load:
                subj_id = f.split('/')[-1] # /home/dir/file.npy
                if 'CAIPI' in subj_id:
                    subj_id = '_'.join(subj_id.split('_')[:-1]) # 1_02_030-V1_CAIPI1x3.npy
                elif 'segEPI' in subj_id:
                    subj_id = subj_id.split('-')
                    if len(subj_id) == 2: subj_id = subj_id[0] + '-V1'
                    elif len(subj_id) == 3: subj_id = subj_id[0] + '-V1-2'
                else:
                    raise NotImplementedError(f'Unrecognized filename: {subj_id}')
                if subj_id in fold_split[args.dataset_type]: keep.append(f)
            print(f'Running model prediction on subjects from fold-{args.fold}:')
            #print(keep)
            files_to_load = keep
        print(f'Found {len(files_to_load)} files at {args.input_path}.')
    else:
        files_to_load = [ args.input_path ]

    for cur_file in tqdm(files_to_load, ncols=80):
        # load data
        print(f'\nLoading: {cur_file}...')

        fname, fext = os.path.splitext(cur_file)
        if 'gz' in fext:
            data = np.array(nib.load(cur_file).dataobj)
        elif 'npy' in fext:
            data = np.load(cur_file)

        if args.type_dir:
            out_name = os.path.join(args.output_path, cur_file.split('/')[-1])
        else:
            out_name = args.output_path

        # prediction stage
        if args.extract_patches:
            patch_start_time = datetime.datetime.now()
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
            patch_end_time = datetime.datetime.now()

            # convert to torch format
            patches = np.moveaxis(patches, -1, 1) # [NPATCHES, 1, PATCH_LEN, PATCH_LEN]
            patches = torch.tensor(patches)
            patches = TensorDataset(patches)
            patches_loader = DataLoader(patches, batch_size=PATCH_BATCH_SIZE)

            # run prediction
            patches_out = []
            for patches_batch in patches_loader:
                patches_batch = patches_batch[0].to(device)
                res = model(
                        patches_batch
                )
                res = res.cpu().detach()
                if res.shape[-1] == 2: res = torch.view_as_complex(res)
                res = res.numpy()
                patches_out.append(res)
            unpatch_start_time = datetime.datetime.now()
            patches_out = np.vstack(patches_out)
            patches_out = np.moveaxis(patches_out, 1, -1)
            patches = np.squeeze(patches_out)
            patches = patches.reshape(patches_shape)
            output = unpatchify(patches, data.shape)
            unpatch_end_time = datetime.datetime.now()
            patch_time, unpatch_time = patch_end_time - patch_start_time, unpatch_end_time - unpatch_start_time
            print(f'Patch time: {patch_time}, Unpatch time: {unpatch_time}')
        else:
            # data.shape 384, 384, 256
            data = np.moveaxis(data, -1, 0) # 256, 384, 384
            data = np.expand_dims(data, axis=3) # 256, 384, 384, 1

            # convert to torch format
            data = np.moveaxis(data, -1, 1) # 256, 1, 384, 384
            data = torch.tensor(data)
            data = TensorDataset(data)
            data_loader = DataLoader(data, batch_size=FULL_BATCH_SIZE)

            # run prediction
            data_out = []
            for i, data_batch in enumerate(data_loader):
                data_batch = data_batch[0].to(device)
                res = model(
                        data_batch
                )
                res = res.cpu().detach()
                if res.shape[-1] == 2: res = torch.view_as_complex(res)
                res = res.numpy()
                data_out.append(res)
            data = np.vstack(data_out)

            data = np.moveaxis(data, 1, -1)
            data = np.squeeze(data)
            output = np.moveaxis(data, 0, -1)

        if np.iscomplexobj(output): save_dtype = 'complex64'
        else:                       save_dtype = 'float32'

        if args.extract_patches and args.debug_patches:
            assert before_patches.shape == patches.shape, f'{before_patches.shape} {patches.shape}'
            before_patches_name = '/'.join(out_name.split('/')[:-1]) + '/before_patches/' + out_name.split('/')[-1]
            after_patches_name = '/'.join(out_name.split('/')[:-1]) + '/after_patches/' + out_name.split('/')[-1]
            write_data(before_patches, before_patches_name, save_dtype, save_format=fext)
            write_data(patches, after_patches_name, save_dtype, save_format=fext)
            print(f'Saving debug patches {after_patches_name}')

        write_data(output, out_name, save_dtype, save_format=fext)
        print(f' Saving: {out_name}')

    print('Complete!')

def create_parser():
    parser = argparse.ArgumentParser()
    example_str = 'python predict_torch.py 2 --type_dir ../data/datasets/accelerated/msrebs_all_testing/{inputs,outputs} ../models/compleximage_full_cdncnn_2022-12-15/complex_dncnn_ep26.h5 384 384'
    parser.add_argument('dimensions', type=int, choices=[2, 3])
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('model_path')
    parser.add_argument('--dataset_type', choices=['train', 'valid', 'test', 'overfit_one', 'train_valid'], default='test')
    parser.add_argument('--debug_patches', action='store_true')
    parser.add_argument('--extract_patches', action='store_true')
    parser.add_argument('--extract_step', nargs='+', type=int, help='[ length width ]')
    parser.add_argument('--fold', type=int, choices=[1,2,3,4,5,6])
    parser.add_argument('--input_size', nargs='+', type=int, help='[ length width ]')
    parser.add_argument('--type_dir', action='store_true', help='input_path/output_path will be intepreted as dirs')

    return parser

if __name__ == '__main__':
    main()

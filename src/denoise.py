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
from preparation.preprocessing_pipeline import rescale_complex, rescale_magnitude, magphase2complex
from preparation.preprocessing_pipeline import pad_square, remove_padding

N_HIDDEN_LAYERS = 12

def main():
    parser = create_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = '_'.join(args.model.split('/')[-1].split('_')[:-1])

    logging.info(f'Model {model_type}: {args.model}')
    logging.info(f'Residual layer: {args.residual_layer}')
    model = get_model(model_type, args.dimensions, N_HIDDEN_LAYERS, args.residual_layer)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to(device)

    out_suffix = f'{model_type}{args.dimensions}D' # cdncnn3D

    # load data
    print(f'Loading {args.input}')
    src_data = nib.load(args.input)
    data = np.array(src_data.dataobj)
    data = rescale_magnitude(data)
    if args.phase:
        print(f'Loading {args.phase}')
        pha_data = np.array(nib.load(args.phase).dataobj)
        data = magphase2complex(data, pha_data, rescale=True)
    else:
        data = data.astype('float32')

    # preprocess data
    # reorder axes for processing
    # pad larger two dimensions to (384, 384)

    data = np.moveaxis(data, args.axis, -1)
    data = pad_square(data)

    # prediction stage
    if args.extract_patches:
        patch_start_time = datetime.datetime.now()
        patch_size, extract_step = args.input_size, args.extract_step
        if args.dimensions == 2:
            if len(patch_size)   == 2: patch_size = patch_size + [1]
            if len(extract_step) == 2: extract_step = extract_step + [1]
        
        # TODO
        # Pad to multiple of 64
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
        patches_loader = DataLoader(patches, batch_size=args.batch_size)

        # run prediction
        patches_out = []
        for patches_batch in tqdm(patches_loader):
            patches_batch = patches_batch[0].to(device)
            res = model(
                    patches_batch
            )
            res = res.cpu().detach()
            if res.shape[-1] == 2: res = torch.view_as_complex(res)
            res = res.numpy()
            patches_out.append(res)
        patches_out = np.vstack(patches_out)
        patches_out = np.moveaxis(patches_out, 1, -1)
        patches = np.squeeze(patches_out)
        patches = patches.reshape(patches_shape)
        output = unpatchify(patches, data.shape)
    else:
        # data.shape 384, 384, 256
        data = np.moveaxis(data, -1, 0) # 256, 384, 384
        data = np.expand_dims(data, axis=3) # 256, 384, 384, 1

        # convert to torch format
        data = np.moveaxis(data, -1, 1) # 256, 1, 384, 384
        data = torch.tensor(data)
        data = TensorDataset(data)
        data_loader = DataLoader(data, batch_size=args.batch_size)

        # run prediction
        data_out = []
        for data_batch in tqdm(data_loader):
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

    output = np.moveaxis(output, -1, args.axis)
    output = remove_padding(output, src_data.shape)

    if np.iscomplexobj(output): save_dtype = 'complex64'
    else:                       save_dtype = 'float32'

    if args.rescale:
        if np.iscomplexobj(output):
            output = rescale_complex(output)
        else:
            output = rescale_magnitude(output)

    if args.phase:
        mag, pha = np.abs(output), np.angle(output)
        nii_mag = nib.Nifti1Image(mag, affine=src_data.affine, header=src_data.header)
        nii_pha = nib.Nifti1Image(pha, affine=src_data.affine, header=src_data.header)
        out_name_mag = args.input.split('.')[0] + '_' + out_suffix + '.nii.gz'
        out_name_pha = args.phase.split('.')[0] + '_' + out_suffix + '.nii.gz'
        nib.save(nii_mag, out_name_mag)
        nib.save(nii_pha, out_name_pha)
        print(f' Saving: {out_name_mag}')
        print(f' Saving: {out_name_pha}')
    else:
        nii = nib.Nifti1Image(output, affine=src_data.affine, header=src_data.header)
        out_name= args.input.split('.')[0] + '_' + out_suffix + '.nii.gz'
        nib.save(nii, out_name)
        print(f' Saving: {out_name}')
    
    print('Complete!')
    return 0

def create_parser():
    parser = argparse.ArgumentParser()
    example_str = 'python predict_torch.py 2 mag.nii.gz -p pha.nii.gz /home/quahb/caipi_denoising/models/compleximage_3d_patches64_fold5_2024-04-07/cdncnn_ep50.pt --input_size 64 64 64 --extract_patches --extract_step 32 32 32 --batch_size 16'
    parser.add_argument('dimensions', type=int, choices=[2, 3])
    parser.add_argument('input')
    parser.add_argument('model')
    parser.add_argument('-a', '--axis', type=int) # dimension 
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('-e', '--extract_patches', action='store_true')
    parser.add_argument('-i', '--input_size', nargs='+', type=int, help='[ length width ]')
    parser.add_argument('-p', '--phase')
    parser.add_argument('-r', '--residual_layer', action='store_true', default=False)
    parser.add_argument('-s', '--extract_step', nargs='+', type=int, help='[ length width ]')
    parser.add_argument('--rescale', action='store_true')

    return parser

if __name__ == '__main__':
    main()

import argparse
import logging
import nibabel as nib
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import pdb
import tensorflow as tf
from tqdm import tqdm
import yaml

from patchify import patchify, unpatchify

from modeling.get_model import get_model
from preparation.data_io import write_data
from preparation.prepare_tf_dataset import np_to_tfdataset


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.dimensions == 2:
        input_shape = [None] + args.input_size + [1]
    elif args.dimensions == 3:
        input_shape = [None] + args.input_size

    network_params = {
        'dimensions': args.dimensions,
        'model_type': args.model_type,
        'input_shape': input_shape,
        'load_model_path': args.model_path
    }

    print(f'Creating model {args.model_type}...')
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = get_model(**network_params)

    if args.type_dir:
        files_to_load = os.listdir(args.input_path)
        print(f'Found {len(files_to_load)} files at {args.input_path} to denoise.')
        files_to_load.sort()
        files_to_load = [ os.path.join(args.input_path, f) for f in files_to_load ]
    else:
        files_to_load = [ args.input_path ]

    for cur_file in tqdm(files_to_load, ncols=80):
        print(f'\nLoading file: {cur_file}...')
        fname, fext = os.path.splitext(cur_file)
        if 'nii' in fext:
            data = np.array(nib.load(cur_file).dataobj)
        elif 'npy' in fext:
            data = np.load(cur_file)

        if args.extract_patches:
            raise NotImplementedError('Predicting on patches not yet implemented correctly for 3D. ~Aug 2023')
            patch_size = args.input_size
            extract_step = args.extract_step
            # extract patches for every slice in subject
            patches_stack = []
            for i in tqdm(range(data.shape[-1]), ncols=80): 
                slc = np.copy(data[:,:,i])
                patches = patchify(slc, patch_size, step=extract_step)
                patchify_shape = patches.shape[:2]
                patches = patches.reshape(-1, *patch_size)
                patches = np.expand_dims(patches, axis=3)
                patches_stack.append(patches)
            patches_stack = np.vstack(patches_stack) # (n_patches, X, Y, 1)
            
            # run prediction
            patches_stack = np_to_tfdataset(patches_stack)
            patches = model.predict(
                    patches_stack,
                    verbose=1,
                    batch_size=32)
            patches = np.squeeze(patches)

            # reconstruct slices from patches
            subj_slices= []
            n_patches_per_slice = len(patches) // len(data.shape[-1])
            patches = patches.reshape(*patchify_shape, *patch_size)
            for slc_i in range(data.shape[-1]):
                slc = unpatchify(patches, data.shape[:-1])
                subj_slices.append(slc)
            denoised_data = np.stack(subj_slices)
            output = np.moveaxis(denoised_data, 0, -1)
        else:
            data = np.moveaxis(data, -1, 0)
            data = np.expand_dims(data, axis=3)
            data = np_to_tfdataset(data)
            data = model.predict(
                    data,
                    verbose=1,
                    batch_size=32)
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
            save_dtype = 'float16'
        write_data(output, out_name, save_dtype)
        print(f'Saving {out_name}')


def create_parser():
    parser = argparse.ArgumentParser()
    example_str = 'python predict.py 3 cdncnn mse --type_dir ../data/datasets/test/msrebs_all_testing/{inputs,outputs} ../models/compleximage_full_cdncnn_2022-12-15/complex_dncnn_ep26.h5 384 384'
    parser.add_argument('dimensions', type=int, choices=[2, 3])
    parser.add_argument('model_type', choices=['dncnn', 'res_dncnn', 'cdncnn'])
    parser.add_argument('loss_function', choices=['mae', 'mse'])
    parser.add_argument('--type_dir', action='store_true', help='input_path/output_path will be intepreted as dirs')
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('model_path')
    parser.add_argument('input_size', nargs='+', type=int, help='[ length width ]')
    parser.add_argument('--extract_patches', action='store_true')
    parser.add_argument('--extract_step', nargs='+', type=int, help='[ length width ]')

    return parser

if __name__ == '__main__':
    main()

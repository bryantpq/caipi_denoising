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
from preparation.prepare_dataset import np_to_tfdataset


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.dimensions in [2, 3]: input_shape = [None] + args.input_size + [1]

    if args.type_dir: out_name = os.path.join(args.output_path, cur_file.split('/')[-1])
    else:             out_name = args.output_path

    if np.iscomplexobj(output): save_dtype = 'complex64'
    else:                       save_dtype = 'float32'

    model_type = args.model_path.split('/')[-1].split('_')[0]

    network_params = {
        'dimensions': args.dimensions,
        'model_type': model_type,
        'input_shape': input_shape,
        'load_model_path': args.model_path
    }

    print(f'Creating model {args.model_type}...')
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = get_model(**network_params)

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
            print(keep)
            files_to_load = keep
        print(f'Found {len(files_to_load)} files at {args.input_path} to denoise.')
    else:
        files_to_load = [ args.input_path ]

    for cur_file in tqdm(files_to_load, ncols=90):
        print(f'\nLoading file: {cur_file}...')
        fname, fext = os.path.splitext(cur_file)
        if 'gz' in fext:
            data = np.array(nib.load(cur_file).dataobj)
        elif 'npy' in fext:
            data = np.load(cur_file)

        if args.extract_patches:
            patch_size, extract_step = args.input_size, args.extract_step
            if args.dimensions == 2:
                if len(patch_size) == 2:
                    patch_size = patch_size + [1]
                if len(extract_step) == 2:
                    extract_step = extract_step + [1]

            patches = patchify(data, patch_size, extract_step)
            patches_shape = patches.shape
            patches = patches.reshape(-1, *patch_size)
            patches = np.expand_dims(patches, axis=-1) # 3D patches: (n_patches, X, Y, Z, 1)
            before_patches = np.copy(patches)

            patches = np_to_tfdataset(patches)
            patches = model.predict(
                    patches,
                    verbose=1,
                    batch_size=32
            )
            patches = np.squeeze(patches)
            patches = patches.reshape(patches_shape)
            output = unpatchify(patches, data.shape)
        else:
            data = np.moveaxis(data, -1, 0)
            data = np.expand_dims(data, axis=3)
            data = np_to_tfdataset(data)
            data = model.predict(
                    data,
                    verbose=1,
                    batch_size=32
            )
            data = np.squeeze(data)
            output = np.moveaxis(data, 0, -1)

        if args.extract_patches and args.debug_patches:
            before_patches_name = '/'.join(out_name.split('/')[:-1]) + '/before_patches/' + out_name.split('/')[-1]
            after_patches_name = '/'.join(out_name.split('/')[:-1]) + '/after_patches/' + out_name.split('/')[-1]
            write_data(before_patches, before_patches_name, save_dtype, save_format=fext)
            write_data(patches, after_patches_name, save_dtype, save_format=fext)
            print(f'Saving debug patches {after_patches_name}')

        write_data(output, out_name, save_dtype, save_format=fext)
        print(f'Saving {out_name}')

    print('Complete!')

def create_parser():
    parser = argparse.ArgumentParser()
    example_str = 'python predict.py 2 cdncnn --type_dir ../data/datasets/accelerated/msrebs_all_testing/{inputs,outputs} ../models/compleximage_full_cdncnn_2022-12-15/cdncnn_ep26.h5 384 384'
    parser.add_argument('dimensions', type=int, choices=[2, 3])
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('model_path')
    parser.add_argument('input_size', nargs='+', type=int, help='[ length width ]')
    parser.add_argument('--type_dir', action='store_true', help='input_path/output_path will be intepreted as dirs')
    parser.add_argument('--extract_patches', action='store_true')
    parser.add_argument('--extract_step', nargs='+', type=int, help='[ length width ]')
    parser.add_argument('--fold', type=int, choices=[1,2,3,4,5])
    parser.add_argument('--debug_patches', action='store_true')

    return parser

if __name__ == '__main__':
    main()

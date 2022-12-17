import argparse
import logging
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pdb
import tensorflow as tf
from tqdm import tqdm
import yaml

from modeling.get_model import get_model
from preparation.prepare_tf_dataset import np_to_tfdataset
from utils.data_io import write_data

from patchify import patchify, unpatchify

def main():
    parser = create_parser()
    args = parser.parse_args()
    config = yaml.safe_load(args.config)

    print(f'Loading file {args.input_file}')
    data = np.load(args.input_file)

    print(f'Creating model {args.model_type}...')
    strategy = tf.distribute.MirroredStrategy(devices=config['gpus'])
    with strategy.scope():
        model = get_model(
                model_type=args.model_type, 
                loss_function=arg.loss_function,
                input_shape=arg.input_size,
                load_model_path=args.model_path)

    if args.extract_patches:
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

    print('Completed denoising...')
    print(f'Saving {args.output_file}')

    save_dtype = 'csingle' if 'complex' in args.model_type else 'float16'
    write_data(output, args.output_file, save_dtype)


def create_parser():
    parser = argparse.ArgumentParser()
#    parser.add_argument('config', type=argparse.FileType('r'))
    parser.add_argument('model_type', choices=['dncnn', 'res_dncnn', 'complex_dncnn'])
    parser.add_argument('loss_function', choices=['mae', 'mse'])
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('model_path')
    parser.add_argument('input_size', nargs=2, type=int, help='[ length width ]')
    parser.add_argument('--extract_patches', action='store_true')
    parser.add_argument('extract_step', nargs=2, type=int, help='[ length width ]')


    return parser

if __name__ == '__main__':
    main()

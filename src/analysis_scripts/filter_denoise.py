import argparse
import datetime
import nibabel as nib
import numpy as np
import os
import pdb
import subprocess
import sys
sys.path.append('/home/quahb/caipi_denoising/src')

from preparation.preprocessing_pipeline import denoise
from tqdm import tqdm

def main():
    parser = create_parser()
    args = parser.parse_args()

    print('Denoising method:', args.method)
    print('Loading:', args.input)
    start = datetime.datetime.now()
    if 'nii' in args.input:
        nii = nib.load(args.input)
        data = np.array(nii.dataobj)
    elif 'npy' in args.input:
        data = np.load(args.input)

    res = denoise(data, args.method)
    end = datetime.datetime.now()

    if 'nii' in args.input:
        res = nib.Nifti1Image(res, affine=nii.affine, header=nii.header)
        res.header.set_data_dtype('float32')
        nib.save(res, args.output)
    elif 'npy' in args.input:
        np.save(args.output, res)

    print('Saving:', args.output)
    print('Time taken:', end - start)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('method', choices=['bm3d', 'nonlocalmeans', 'nlm'])

    return parser

if __name__ == '__main__':
    main()

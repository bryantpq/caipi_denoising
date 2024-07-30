import argparse
import nibabel as nib
import numpy as np
import os
import pdb
import sys
import subprocess

sys.path.append('/home/quahb/caipi_denoising/src')

from pathlib import Path
from preparation.preprocessing_pipeline import rescale_magnitude, rescale_complex
from tqdm import tqdm

def main():
    parser = create_parser()
    args = parser.parse_args()

    nii = nib.load(args.file)
    data = np.array(nii.dataobj)

    if args.mode == 'rescale':
        if args.type == 'magnitude':
            rescale_data = rescale_magnitude(data, 0, 1)
        elif args.type == 'phase':
            rescale_data = rescale_magnitude(data, -np.pi, np.pi)
        else:
            rescale_data = rescale_complex(data)
    elif args.mode == 'clip':
        if args.type == 'magnitude':
            rescale_data = np.clip(data, 0, 1)
        else:
            raise NotImplementedError()

    out_nii = nib.Nifti1Image(rescale_data, affine=nii.affine, header=nii.header)
    nib.save(out_nii, args.file)

def create_parser():
    parser = argparse.ArgumentParser()

    # ~/caipi_denoising/data/niftis/cavsms_analysis/1_01_016-V1/file.nii.gz
    parser.add_argument('file')
    parser.add_argument('type', choices=['magnitude', 'phase', 'complex'])
    parser.add_argument('--mode', choices=['rescale', 'clip'], default='rescale')

    return parser

if __name__ == '__main__':
    main()

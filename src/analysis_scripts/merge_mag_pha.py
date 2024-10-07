import argparse
import nibabel as nib
import numpy as np
import os
import pdb
import sys
import subprocess

sys.path.append('/home/quahb/caipi_denoising/src')

from pathlib import Path
from preparation.preprocessing_pipeline import magphase2complex
from tqdm import tqdm

def main():
    parser = create_parser()
    args = parser.parse_args()

    mag_nii = nib.load(args.mag)
    mag_data = np.array(mag_nii.dataobj)
    pha_nii = nib.load(args.pha)
    pha_data = np.array(pha_nii.dataobj)

    complex_data = magphase2complex(mag_data, pha_data)

    mag_nii.header.set_data_dtype(np.complex64)
    out_nii = nib.Nifti1Image(complex_data, affine=mag_nii.affine, header=mag_nii.header)
    nib.save(out_nii, args.out_name)

def create_parser():
    parser = argparse.ArgumentParser()

    # ~/caipi_denoising/data/niftis/cavsms_analysis/1_01_016-V1/file.nii.gz
    parser.add_argument('mag')
    parser.add_argument('pha')
    parser.add_argument('out_name')

    return parser

if __name__ == '__main__':
    main()

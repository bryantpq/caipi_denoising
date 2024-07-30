import argparse
import nibabel as nib
import numpy as np
import os
import pdb
import subprocess

from pathlib import Path
from tqdm import tqdm

def main():
    parser = create_parser()
    args = parser.parse_args()

    nii = nib.load(args.mask)
    data = np.array(nii.dataobj)

    classes = {
        'tha': [49, 10],
        'cau': [50, 11],
        'put': [51, 12],
        'glo': [52, 13],
        'wm_nonuc' : [41, 2]
    }

    for k in classes.keys():
        c1, c2 = classes[k]
        mask = np.zeros(data.shape)
        for x,y,z in zip(*np.where(data == c1)):
            mask[x,y,z] = 1
        for x,y,z in zip(*np.where(data == c2)):
            mask[x,y,z] = 1
#        mask1 = np.ma.masked_where(data == c1, data)
#        mask2 = np.ma.masked_where(data == c2, data)
        out_nii = nib.Nifti1Image(mask, affine=nii.affine, header=nii.header)
        out_name = os.path.join(os.path.dirname(args.mask), f'{k}_mask.nii.gz')
        print(f'Saving {out_name}')
        nib.save(out_nii, out_name)

def create_parser():
    parser = argparse.ArgumentParser()

    # ~/caipi_denoising/data/niftis/cavsms_analysis/1_01_016-V1/file.nii.gz
    parser.add_argument('mask')

    return parser

if __name__ == '__main__':
    main()

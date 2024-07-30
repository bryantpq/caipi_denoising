import argparse
import nibabel as nib
import numpy as np
import os
import pdb
import subprocess

def main():
    parser = create_parser()
    args = parser.parse_args()

    # load data
    nii = nib.load(args.input)
    data = np.array(nii.dataobj)

    # process
    res = data
    res = np.transpose(res, [2,1,0])
    res = res[36:384-36, :, 64: 384-64]
    res = np.flip(res, 0)

    # Rescale phase data
    subj = args.input.split('/')[-2]
    ref = f'/home/quahb/caipi_denoising/data/niftis/cavsms_analysis/{subj}/3D_T2STAR_segEPI.nii.gz'
    ref = nib.load(ref)
    res = nib.Nifti1Image(res, affine=ref.affine, header=ref.header)
    res.header.set_data_dtype(data.dtype)
    nib.save(res, args.output)
    print('Saving output to', args.output)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')

    return parser

if __name__ == '__main__':
    main()

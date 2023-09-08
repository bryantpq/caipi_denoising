import argparse
import nibabel as nib
import numpy as np
import os
import pdb
import scipy.io

from tqdm import tqdm


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.path != False:
        if not os.path.isdir(args.path):
            print('Given path is not a dir: ' + args.path)
            print('Exiting...')
            return

    files = os.listdir(args.path)
    files = [ f for f in files if '.npy' in f or '.nii.gz' in f]
    print('Found {} files at {}'.format(len(files), args.path))
    files = [ os.path.join(args.path, f) for f in files ]

    for f in tqdm(files, ncols=100):
        print()
        print('Loading file: {}'.format(f))

        file_pref = f.split('/')[-1].split('.')[0] # will capture filename before extension without dir path
        file_ext = '.'.join(f.split('/')[-1].split('.')[1:]) # will capture 'npy' or 'nii.gz'
        if file_ext == 'npy':
            data = np.load(f)
        elif file_ext == 'nii.gz':
            data = np.array(nib.load(f).dataobj)

        assert np.iscomplexobj(data), 'Given data should have complex numbers'

        mag, pha = np.abs(data), np.angle(data)
        mag_fname = os.path.join(args.path, f'{file_pref}_mag.{file_ext}')
        pha_fname = os.path.join(args.path, f'{file_pref}_pha.{file_ext}')

        print(f'Saving file: {mag_fname}')
        print(f'Saving file: {pha_fname}')

        if file_ext == 'npy':
            np.save(mag_fname, mag)
            np.save(pha_fname, pha)
        elif file_ext == 'nii.gz':
            mag_nii = nib.Nifti1Image(mag, affine=np.eye(4))
            pha_nii = nib.Nifti1Image(pha, affine=np.eye(4))
            nib.save(mag_nii, mag_fname)
            nib.save(pha_nii, pha_fname)

    print('Complete!')

def create_parser():
    parser = argparse.ArgumentParser()
    example_str = 'python split_magpha.py /path/to/dir'
    parser.add_argument('path', default=False)

    return parser

if __name__ == '__main__':
    main()

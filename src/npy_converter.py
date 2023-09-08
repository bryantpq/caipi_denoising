import argparse
import nibabel as nib
import numpy as np
import os
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

    if args.format in ['nifti', 'nii']:
        RESULT_DIR = 'nii'
        FILE_EXT = 'nii.gz'
    elif args.format == 'mat':
        RESULT_DIR = 'mat'
        FILE_EXT = 'mat'

    files = os.listdir(args.path)
    files = [ f for f in files if '.npy' in f ]
    print('Found {} files at {}'.format(len(files), args.path))
    files = [ os.path.join(args.path, f) for f in files ]
    RESULT_DIR = os.path.join(args.path, RESULT_DIR)

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    for f in tqdm(files, ncols=100):
        print('Loading file: {}'.format(f))
        data = np.load(f)
        fname = f.split('/')[-1]
        fname = fname[:-3] + FILE_EXT
        fname = os.path.join(RESULT_DIR, fname)
        print('Saving file: {}'.format(fname))

        if args.format in ['nifti', 'nii']:
            nii_data = nib.Nifti1Image(data, affine=np.eye(4))
            nib.save(nii_data, fname)
        elif args.format == 'mat':
            scipy.io.savemat(fname, dict(x=data))

    print('Complete!')

def create_parser():
    parser = argparse.ArgumentParser()
    example_str = 'python npy_converter.py {nii, mat} /path/to/dir'
    parser.add_argument('format', choices=['nifti', 'nii', 'mat'])
    parser.add_argument('path', default=False)

    return parser

if __name__ == '__main__':
    main()

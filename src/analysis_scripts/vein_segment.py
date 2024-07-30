import argparse
import nibabel as nib
import numpy as np
import os
import pdb
import subprocess
import time

from tqdm import tqdm

def main():
    parser = create_parser()
    args = parser.parse_args()

    # 1. get brain mask
    args.bet = os.path.abspath(args.bet)
    bet_file = args.bet.split('.')[0] + '_bet.nii.gz'
    print('Creating bet file:', bet_file, '...', end=' ')
    result = subprocess.run(
            ['bet', args.bet, bet_file]
    )
    print('Done!')
    if result.stderr is not None: print(result.stderr)

    # 2. generate vein mask
    print('Generating vein mask...')
    outpath = '/'.join(args.bet.split('/')[:-1])
    outname = bet_file.split('.')[0] + f'_vein_th{args.threshold}'

    matlab_fn = f"Xcr_Frangi3Dveinmask('{bet_file}', '{outpath}', '{outname}', '{args.threshold}')"
    print(matlab_fn)

    SCRIPT_DIR = '/home/quahb/caipi_denoising/frangi_filter_version2a/'
    result = subprocess.run(
            ['matlab', '-batch', matlab_fn],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True
    )
    #print(result.stdout)
    print(result.stderr)

    # 3. remove brain mask
    print('Removing bet file:', bet_file)
    result = subprocess.run(
            ['rm', bet_file]
    )
    if result.stderr is not None: print(result.stderr)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('bet')
    parser.add_argument('threshold', type=float, default=0.01)

    return parser

if __name__ == '__main__':
    main()

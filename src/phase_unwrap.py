import argparse
import nibabel as nib
import numpy as np
import os
import pdb
import subprocess

from tqdm import tqdm

def main():
    parser = create_parser()
    args = parser.parse_args()

    assert 'complex' in args.experiment, 'This script should only be ran for complex valued data.'

    files = [ os.path.join(args.path, f) for f in os.listdir(args.path) ]
    mag_files = sorted([ f for f in files if os.path.isfile(f) and 'Mag' in f ])
    pha_files = sorted([ f for f in files if os.path.isfile(f) and 'Pha' in f ])
    assert len(mag_files) == len(pha_files)

    BET_DIR = f'/home/quahb/caipi_denoising/data/datasets/analysis/brain_mask/{args.experiment}/{args.target_folder}'
    PHASE_DIR = f'/home/quahb/caipi_denoising/data/datasets/analysis/phase_unwrap/{args.experiment}/{args.target_folder}'
    SCRIPT_DIR = '/home/quahb/caipi_denoising/STISuite_V3.0/'
    strip_path_fext = lambda x: x.split('/')[-1].split('.')[0]

    if not os.path.exists(BET_DIR):   os.makedirs(BET_DIR)
    if not os.path.exists(PHASE_DIR): os.makedirs(PHASE_DIR)

    print(f'Found {len(mag_files)} files at {args.path}')    
    for mag, pha in tqdm(zip(mag_files, pha_files), ncols=90, total=len(mag_files)):
        mag_, pha_ = map( strip_path_fext, (mag, pha) )
        mag_, pha_ = '_'.join(mag_.split('_')[:-1]), '_'.join(pha_.split('_')[:-1])
        assert mag_ == pha_, f'Different file names {mag_} {pha_}'
        print()
        print(f'Processing {mag_} ... ', end='', flush=True)

        # run brain extraction
        mag_file = mag.split('/')[-1]
        bet_out = os.path.join(BET_DIR, mag_file)
        result = subprocess.run(
                ['bet', mag, bet_out],
                capture_output=True,
                text=True
        )
        #print(result.stdout)
        print(result.stderr)
        assert os.path.exists(bet_out), f'Error: Output not created for {bet_out}'

        pha_subj = strip_path_fext(pha)
        matlab_fn = f"Gen_tissue_phase('{bet_out}', '{pha}', '{PHASE_DIR}', '{pha_subj}')"

        # run phase unwrapping
        result = subprocess.run(
                ['matlab', '-batch', matlab_fn],
                cwd=SCRIPT_DIR,
                capture_output=True,
                text=True
        )
        #print(result.stdout)
        print(result.stderr)
        pha_out = os.path.join(PHASE_DIR, pha_subj + '.nii.gz')
        assert os.path.exists(pha_out), f'Error: Output not created for {pha_out}'
        print(f'Completed!')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment')
    parser.add_argument('target_folder')
    parser.add_argument('path')

    return parser

if __name__ == '__main__':
    main()

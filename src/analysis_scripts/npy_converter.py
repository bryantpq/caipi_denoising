import argparse
import nibabel as nib
import numpy as np
import os
import pdb
import scipy.io
import shutil
import sys
sys.path.append('/home/quahb/caipi_denoising/src')

from multiprocessing import Process
from tqdm import tqdm

from utils.standardize_nifti import standardize_affine_header

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
    files = sorted([ os.path.join(args.path, f) for f in files ])
    RESULT_DIR = os.path.join(args.path, RESULT_DIR)

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if args.postprocess: print('Postprocessing data...')
    if args.split_mag_pha: print('Splitting magnitude and phase data...')

    def task(f):
        print()
        print('Loading file: {}'.format(f))
        data = np.load(f)
        fname = f.split('/')[-1] 
        fname = fname[:-3] + FILE_EXT
        fname = os.path.join(RESULT_DIR, fname)
        print('Saving file: {}'.format(fname))

        if args.study == 'cavsms':
            acceleration = f.split('/')[-1].split('_')[3].split('.')[0]
            if acceleration == 'rNC':
                acceleration = f.split('/')[-1].split('_')[-1].split('.')[0] # m2x2Warped
                acceleration = 'CAIPI' + acceleration[1:4]
            subject_id = '_'.join(f.split('/')[-1].split('_')[:3])
            if '3D' in acceleration: acceleration = '3D_T2STAR_segEPI'

        elif args.study == 'msrebs':
            acceleration = f.split('/')[-1].split('_')[1].split('.')[0]
            subject_id = '_'.join(f.split('/')[-1].split('_')[:1])

        if args.postprocess:
            if args.study == 'cavsms':
                data = data[36:384-36, :, :]
            elif args.study == 'msrebs':
                data = data[12:384-12, :, :]
                data = np.flip(data, axis=1)

        if args.format in ['nifti', 'nii']:
            nii_data = standardize_affine_header(data, subject_id, acceleration, args.study)
            nib.save(nii_data, fname)

            if args.split_mag_pha and np.iscomplexobj(data):
                mag_data, pha_data = np.abs(data), np.angle(data)
                mag_nii = standardize_affine_header(mag_data, subject_id, acceleration, args.study)
                pha_nii = standardize_affine_header(pha_data, subject_id, acceleration, args.study)

                mag_fname = fname.split('.')[0] + '_Mag.' + '.'.join(fname.split('.')[1:])
                pha_fname = fname.split('.')[0] + '_Pha.' + '.'.join(fname.split('.')[1:])
                nib.save(mag_nii, mag_fname)
                nib.save(pha_nii, pha_fname)

        elif args.format == 'mat':
            scipy.io.savemat(fname, dict(x=data))

    processes = [ Process(target=task, args=(f,)) for f in files ]

    for p in processes: p.start()
    for p in processes: p.join()
    
    print('Done processing', flush=True)

    exp, out_folder = RESULT_DIR.split('/')[-3:-1]
    ANALYSIS_PATH = f'/home/quahb/caipi_denoising/data/datasets/analysis/denoised/{exp}/{out_folder}'
    print(f'Copying {RESULT_DIR} to {ANALYSIS_PATH}...')
    if os.path.exists(ANALYSIS_PATH):
        shutil.rmtree(ANALYSIS_PATH)
        shutil.copytree(RESULT_DIR, ANALYSIS_PATH)

    print('Complete!')

def create_parser():
    parser = argparse.ArgumentParser()
    example_str = 'python npy_converter.py {nii, mat} /path/to/dir'
    parser.add_argument('format', choices=['nifti', 'nii', 'mat'])
    parser.add_argument('study', choices=['cavsms', 'msrebs'])
    parser.add_argument('--postprocess', action='store_true')
    parser.add_argument('--split_mag_pha', action='store_true')
    parser.add_argument('path', default=False)

    return parser

if __name__ == '__main__':
    main()

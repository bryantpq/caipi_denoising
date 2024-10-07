import argparse
import os
import pdb
import shutil
import sys

from multiprocessing import Process
from tqdm import tqdm

def main():
    parser = create_parser()
    args = parser.parse_args()

    exp, out_folder, __ = args.experiment.split('/')[-4:-1]
    assert __ == 'nii', 'Path must end with nii/'

    files = [ os.path.join(args.experiment, f) for f in os.listdir(args.experiment) ]

    analysis_folder = '/home/quahb/caipi_denoising/data/niftis/cavsms_analysis'

    def task(f):
        if '1_01_037-V1-2' in f:
            return
        subj = get_subj(f)
        if args.out_folder is not None:
            dst = os.path.join(analysis_folder, subj, 'denoised', args.out_folder)
        elif out_folder == 'outputs':
            dst = os.path.join(analysis_folder, subj, 'denoised', exp)
        else:
            dst = os.path.join(analysis_folder, subj, 'denoised', '{}_{}'.format(exp, out_folder))
        os.makedirs(dst, exist_ok=True)

        print(f'Copying {f}   ->   {dst}')
        shutil.copy2(f, dst)

    processes = [ Process(target=task, args=(f,)) for f in files ]

    for p in processes: p.start()
    for p in processes: p.join()
    
    print('Done processing', flush=True)


def get_subj(full_path_file):
    file_name = full_path_file.split('/')[-1]

    if len(file_name.split('-')) == 2: # 1_01_037-V1
        subj = file_name.split('-')[0] + '-V1'
    elif len(file_name.split('-')) == 3: # 1_01_037-V1-2
        subj = file_name.split('-')[0] + '-V1-2'

    return subj

def create_parser():
    parser = argparse.ArgumentParser()
    # /home/quahb/caipi_denoising/data/datasets/accelerated/magnitude_3d_patches64/outputs/nii/
    parser.add_argument('experiment', help='string must end with /')
    parser.add_argument('--out_folder')

    return parser


if __name__ == '__main__':
    main()

import argparse
import nibabel as nib
import numpy as np
import os
import pdb
import subprocess

from preparation.preprocessing_pipeline import rescale_magnitude
from tqdm import tqdm

def main():
    '''
    Target folder should have Mag and Pha files
    '''
    parser = create_parser()
    args = parser.parse_args()

    files = [ os.path.join(args.path, f) for f in os.listdir(args.path) ]

    # Grab only the phase files
    no_caipi_files = sorted([ f for f in files if 'segEPI_pha.nii.gz' in f ])
    before_files = sorted([ f for f in files if '1x2_pha.nii.gz' in f or '1x3_pha.nii.gz' in f or '2x2_pha.nii.gz' in f ])
    after_files = sorted([ f for f in files if 'Pha.nii.gz' in f ])
    # Grab the magnitude brain extracted mask
    bet_files = sorted([ f for f in files if 'bet' in f ])

    assert len(before_files) == len(after_files)

    SCRIPT_DIR = '/home/quahb/caipi_denoising/STISuite_V3.0/'
    strip_path_fext = lambda x: x.split('/')[-1].split('.')[0]

    to_unwrap = [*no_caipi_files, *before_files, *after_files]
    print(f'Processing phase files:\n{to_unwrap}\n')

    pbar = tqdm(to_unwrap)
    for phase_file in pbar:
        pbar.set_description(f'Unwrapping {phase_file}')

        # Grab corresponding brain extracted file
        if 'CAIPI' not in phase_file:
            bet_file = bet_files[0]
        elif 'CAIPI' in phase_file:
            if '1x2' in phase_file:
                match = [b for b in bet_files if '1x2' in b]
            elif '1x3' in phase_file:
                match = [b for b in bet_files if '1x3' in b]
            elif '2x2' in phase_file:
                match = [b for b in bet_files if '2x2' in b]
            bet_file = match[0]
        else:
            raise NotImplementedError()

        # Rescale phase data
        phase_nii = nib.load(phase_file)
        phase_data = np.array(phase_nii.dataobj)
        phase_data = rescale_magnitude(phase_data, -np.pi, np.pi)
        phase_data = nib.Nifti1Image(phase_data, affine=phase_nii.affine, header=phase_nii.header)
        phase_data.header.set_data_dtype('float32')
        nib.save(phase_data, phase_file)

        pha_subj = strip_path_fext(phase_file) + '_unwrap'
        matlab_fn = f"Gen_tissue_phase('{bet_file}', '{phase_file}', '{args.path}', '{pha_subj}')"
        print(matlab_fn)

        # Run phase unwrapping
        result = subprocess.run(
                ['matlab', '-batch', matlab_fn],
                cwd=SCRIPT_DIR,
                capture_output=True,
                text=True
        )
        #print(result.stdout)
        print(result.stderr)
        print(f'Completed {pha_subj}')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')

    return parser

if __name__ == '__main__':
    main()

import argparse
import nibabel as nib
import numpy as np
import os
import pdb
import subprocess

from preparation.preprocessing_pipeline import rescale_magnitude
from tqdm import tqdm

def main():
    parser = create_parser()
    args = parser.parse_args()

    # Rescale phase data
    phase_nii = nib.load(args.phase)
    phase_data = np.array(phase_nii.dataobj)
    phase_data = rescale_magnitude(phase_data, -np.pi, np.pi)
    phase_data = nib.Nifti1Image(phase_data, affine=phase_nii.affine, header=phase_nii.header)
    phase_data.header.set_data_dtype('float32')
    nib.save(phase_data, args.phase)

    outpath = '/'.join(args.phase.split('/')[:-1])
    outname = args.phase.split('/')[-1].split('.')[0] + '_unwrap' # /dir/path/file.nii.gz -> file_unwrap

    matlab_fn = f"Gen_tissue_phase('{args.bet}', '{args.phase}', '{outpath}', '{outname}')"
    print(matlab_fn)

    # Run phase unwrapping
    SCRIPT_DIR = '/home/quahb/caipi_denoising/STISuite_V3.0/'
    result = subprocess.run(
            ['matlab', '-batch', matlab_fn],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True
    )
    #print(result.stdout)
    print(result.stderr)

    res = os.path.join(outpath, outname + '.nii.gz')
    out_nii = nib.load(res)
    out_data = np.array(out_nii.dataobj)
    out_data = nib.Nifti1Image(out_data, affine=phase_nii.affine, header=phase_nii.header)
    out_data.header.set_data_dtype('float32')
    nib.save(out_data, res)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('bet')
    parser.add_argument('phase')

    return parser

if __name__ == '__main__':
    main()

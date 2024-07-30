import argparse
import numpy as np
import nibabel as nib
import os

def main():
    parser = create_parser()
    args = parser.parse_args()
    args.mask = os.path.abspath(args.mask)

    threshold_mask(args.mask, th=args.threshold, mode=args.mode, debug=True)

def threshold_mask(f, th=0.7, mode='greater', debug=False):
    f = str(f)
    data = nib.load(f)
    d = np.array(data.dataobj)

    if mode == 'greater':
        mask = np.array(d > th, dtype=np.uint8)
    elif mode == 'less':
        mask = np.array(d > th, dtype=np.uint8)
    elif mode == 'non':
        mask = np.array(d != th, dtype=np.uint8)
    elif mode == 'range':
        assert th[0] < th[1], print(f'Expected first arg to be less than second arg')
        m1, m2 = np.zeros(data.shape), np.zeros(data.shape)
        m1[np.where(th[0] < d)] = 1
        m2[np.where(th[0] < d)] = 1

        mask = np.multiply(m1, m2)
        mask = mask.astype('uint8')

    nii = nib.Nifti1Image(mask, affine=data.affine, header=data.header)
    fname = f.split('/')[-1] # probability_map.nii.gz
    fname = f.split('.')[0] # probability_map
    fname = fname + f'_th{th}.nii.gz'
    fname = ''.join(fname.split())
    fname = fname.replace('[', '')
    fname = fname.replace(']', '')

    if debug:
        print(f'Saving {fname}')

    nib.save(nii, fname)

def reorient(f, axis=[], debug=False):
    f = str(f)
    data = nib.load(f)
    d = np.array(data.dataobj)
    reoriented = np.flip(d, axis=axis)

    subj = f.split('/')[-2]
    ref_nii = f'/home/quahb/caipi_denoising/data/niftis/cavsms_analysis/{subj}/3D_T2STAR_segEPI.nii.gz'
    ref_nii = nib.load(ref_nii)

    nii = nib.Nifti1Image(reoriented, affine=ref_nii.affine, header=ref_nii.header)

    nib.save(nii, f)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('mask')
    parser.add_argument('--mode', choices=['less', 'greater', 'non', 'range'], default='greater')
    parser.add_argument('threshold', type=float, nargs='+')

    return parser

if __name__ == '__main__':
    main()

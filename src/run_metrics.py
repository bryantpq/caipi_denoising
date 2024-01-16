import argparse
import nibabel as nib
import numpy as np
import pdb
import os

from pathlib import Path

from evaluation.compute_metrics import axial_psnr, ssim

PRECISION = 4
MASK_PATH = '/home/quahb/caipi_denoising/data/niftis/cavsms_masks/'
MASK_THRESHOLD = 0.0

def main():
    parser = create_parser()
    args = parser.parse_args()

    input_files  = sorted(os.listdir(args.input_path))
    output_files = sorted(os.listdir(args.output_path))

    input_files  = [ os.path.join(args.input_path, f)  for f in input_files if '.nii.gz' in f ]
    output_files = [ os.path.join(args.output_path, f) for f in output_files if '.nii.gz' in f ]

    assert len(input_files) == len(output_files)

    if args.data_type in ['mag', 'magnitude']:
        args.data_type = 'magnitude'
    elif args.data_type in ['complex', 'compleximage']:
        args.data_type = 'compleximage'
    else:
        raise ValueError(f'Unknown data_type: {args.data_type}')
    
    # put data into tuple: (input, output, mask)
    mask_files = []
    masks = list(Path(MASK_PATH).rglob('*/wm_mask.nii.gz'))
    masks = [ str(f) for f in masks ]
    input_output_mask = []
    for in_f, out_f in zip(input_files, output_files):
        # /inputs/nii/1_08_049-V1_CAIPI1x2.nii.gz
        data_subj = in_f.split('/')[-1] 
        for mask in masks:
            # /niftis/cavsms_masks/1_07_003-V1/wm_mask.nii.gz
            mask_subj = mask.split('/')[-2] 
            if mask_subj in data_subj:
                input_output_mask.append( [in_f, out_f, mask] )
                break
    assert len(input_files) == len(input_output_mask)

    res = {}
    # 'CAIPI1x2': 'Subj1': [before_psnr, after_psnr, ssim]

    for in_f, out_f, mask_f in input_output_mask:
        in_subj = in_f.split('/')[-1]
        out_subj = out_f.split('/')[-1]
        acceleration = in_f.split('_')[-1].split('.')[0]

        assert in_subj == out_subj, f'Found mismatching file names {in_subj}, {out_subj}'
        subj_id = in_subj

        in_data, out_data = np.array(nib.load(in_f).dataobj), np.array(nib.load(out_f).dataobj)
        mask_data = np.array(nib.load(mask_f).dataobj)
        mask_data = np.array(mask_data > MASK_THRESHOLD, dtype=np.uint8)

        assert in_data.shape    == (312, 384, 256), f'in_data.shape: {in_data.shape}'
        assert out_data.shape   == (312, 384, 256), f'out_data.shape: {out_data.shape}'
        assert mask_data.shape  == (312, 384, 256), f'mask_data.shape: {mask_data.shape}'

        # compute metrics
        before_psnr = axial_psnr(in_data,  mask=mask_data, subj_id=subj_id)
        after_psnr  = axial_psnr(out_data, mask=mask_data, subj_id=subj_id)

        if args.data_type == 'magnitude':
            #res_ssim = ssim(in_data, out_data)
            res_ssim = 0
        elif args.data_type == 'compleximage':
            res_ssim = ssim(np.abs(in_data), np.abs(out_data))

        res_metrics = [ before_psnr, after_psnr, res_ssim ]
        res_metrics = [ round(m, PRECISION) for m in res_metrics ]

        print(subj_id, res_metrics)

        # add placeholders in dictionary
        if acceleration not in res: res[acceleration] = {}
        if subj_id not in res[acceleration]: res[acceleration][subj_id] = []

        # update values in dictionary
        res[acceleration][subj_id] = res_metrics

    return

def create_parser():
    parser = argparse.ArgumentParser()
    example_str = 'python run_metrics.py magnitude /path/to/{inputs,outputs}'
    parser.add_argument('data_type', choices=['magnitude', 'mag', 'compleximage', 'complex'])
    parser.add_argument('input_path')
    parser.add_argument('output_path')

    return parser

if __name__ == '__main__':
    main()

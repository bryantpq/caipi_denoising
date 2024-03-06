import argparse
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pdb
import os

from pathlib import Path

from evaluation.metrics import snr, psnr, msnr, ssim, mask_intersect_cylinder, erode_mask
from preparation.preprocessing_pipeline import magphase2complex

PRECISION = 4
UNACCELERATED_PATH = '/home/quahb/caipi_denoising/data/niftis/cavsms/'
MASK_PATH = '/home/quahb/caipi_denoising/data/niftis/cavsms_masks/'
MASK_THRESHOLD = 0.0

def main():
    parser = create_parser()
    args = parser.parse_args()

    config = args.input_path.split('/')[-2]
    input_files  = sorted(os.listdir(args.input_path))

    if args.output_path is not None:
        output_files = sorted(os.listdir(args.output_path))

    # TODO
    # output should be optional, only if two, then compute SSIM

    if args.file_type == 'nii':
        input_files  = [ os.path.join(args.input_path, f)  for f in input_files if '.nii.gz' in f ]
        output_files = [ os.path.join(args.output_path, f) for f in output_files if '.nii.gz' in f ]

        assert len(input_files)  > 0, f'No input files found at {args.input_path}. Check that .npy files are converted to .nii.gz format.'
        assert len(output_files) > 0, f'No output files found at {args.output_path}. Check that .npy files are converted to .nii.gz format.'
    elif args.file_type == 'npy':
        input_files  = [ os.path.join(args.input_path, f)  for f in input_files if '.npy' in f ]
        output_files = [ os.path.join(args.output_path, f) for f in output_files if '.npy' in f ]

        assert len(input_files)  > 0, f'No input files found at {args.input_path}. Check that .npy files are converted to .nii.gz format.'
        assert len(output_files) > 0, f'No output files found at {args.output_path}. Check that .npy files are converted to .nii.gz format.'

    assert len(input_files) == len(output_files), f'{len(input_files)}, {len(output_files)}'

    if args.data_type in ['mag', 'magnitude']:
        args.data_type = 'magnitude'
    elif args.data_type in ['complex', 'compleximage']:
        args.data_type = 'compleximage'
    else:
        raise ValueError(f'Unknown data_type: {args.data_type}')
    
    masks = list(Path(MASK_PATH).rglob('*/wm_mask.nii.gz'))
    masks = [ str(f) for f in masks ]

    # 1. Calculate metrics for unaccelerated
    if args.unaccelerated:
        print('Calculating unaccelerated...')
        unaccelerated = list(Path(UNACCELERATED_PATH).rglob('*/3D_T2STAR_segEPI.nii.gz'))
        unaccelerated = sorted([ str(f) for f in unaccelerated ])

        unaccelerated_mask = []
        for unacc in unaccelerated:
            data_subj = unacc.split('/')[-2]
            for mask in masks:
                mask_subj = mask.split('/')[-2]
                if mask_subj in data_subj:
                    unaccelerated_mask.append( [unacc, mask] )
                    break

        res = {}
        for unacc_f, mask_f in unaccelerated_mask:
            subj_id = unacc_f.split('/')[-2]

            unacc_data, mask_data = np.array(nib.load(unacc_f).dataobj), np.array(nib.load(mask_f).dataobj)
            assert unacc_data.shape == (312, 384, 256)
            mask_data = np.array(mask_data > MASK_THRESHOLD, dtype=np.uint8)
            mask_data = mask_intersect_cylinder(mask_data)

            if args.data_type == 'compleximage':
                assert 'complex' in config
                UPHA_DIR = f'/home/quahb/caipi_denoising/data/datasets/analysis/phase_unwrap/complex_unacc/processed'
                upha_data = os.path.join(UPHA_DIR, f'{subj_id}_3D_T2STAR_segEPI_Pha.nii.gz')
                upha_data = np.array(nib.load(upha_data).dataobj)

                assert upha_data.shape == (312, 384, 256)
                if args.save_phase_combination:
                    np.save(f'/home/quahb/caipi_denoising/tmp/{subj_id}_unacc_before.npy', unacc_data)

                unacc_data = magphase2complex(unacc_data, upha_data, rescale=True)

                if args.save_phase_combination:
                    np.save(f'/home/quahb/caipi_denoising/tmp/{subj_id}_unacc_after.npy', unacc_data)

            unacc_psnr = psnr(unacc_data, mask=mask_data, subj_id=subj_id)
            unacc_msnr = msnr(unacc_data, mask=mask_data, subj_id=subj_id)
            unacc_snr  = snr(unacc_data, mask=mask_data)

            res_metrics = [ unacc_psnr, unacc_msnr, unacc_snr ]
            res_metrics = [ round(m, PRECISION) for m in res_metrics ]

            res_metrics = {
                'psnr': res_metrics[0],
                'msnr': res_metrics[1],
                'snr':  res_metrics[2]
            }

            print(subj_id, end=' ')
            for k in res_metrics.keys():
                print(k, res_metrics[k], end=' ')
            print()

    # 2. Calculate metrics for accelerated before/after
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

    res = {}
    # 'CAIPI1x2': 'Subj1': [before_psnr, after_psnr, ssim]

    reset_plt = 0
    for in_f, out_f, mask_f in input_output_mask:
        in_subj = in_f.split('/')[-1]
        out_subj = out_f.split('/')[-1]
        acceleration = in_f.split('_')[-1].split('.')[0]

        assert in_subj == out_subj, f'Found mismatching file names {in_subj}, {out_subj}'
        subj_id = '_'.join(in_subj.split('_')[:-1])

        if args.file_type == 'nii':
            in_data, out_data = np.array(nib.load(in_f).dataobj), np.array(nib.load(out_f).dataobj)
        elif args.file_type == 'npy':
            in_data, out_data = np.load(in_f), np.load(out_f)

        if in_data.shape == (384, 384, 256) and out_data.shape == (384, 384, 256):
            no_pad = slice(36, 384 - 36)
            in_data  =  in_data[no_pad,:,:]
            out_data = out_data[no_pad,:,:]

        mask_data = np.array(nib.load(mask_f).dataobj)
        mask_data = np.array(mask_data > MASK_THRESHOLD, dtype=np.uint8)
        mask_data = mask_intersect_cylinder(mask_data)

        assert in_data.shape   == (312, 384, 256), f'in_data.shape: {in_data.shape}'
        assert out_data.shape  == (312, 384, 256), f'out_data.shape: {out_data.shape}'
        assert mask_data.shape == (312, 384, 256), f'mask_data.shape: {mask_data.shape}'

        # merge magnitude and complex
        if args.data_type == 'compleximage':
            assert 'complex' in config
            UPHA_DIR = f'/home/quahb/caipi_denoising/data/datasets/analysis/phase_unwrap/{config}'
            input_upha  = os.path.join(UPHA_DIR, f'inputs/{subj_id}_{acceleration}_Pha.nii.gz')
            output_upha = os.path.join(UPHA_DIR, f'outputs/{subj_id}_{acceleration}_Pha.nii.gz')
            input_upha  = np.array(nib.load(input_upha).dataobj)
            output_upha = np.array(nib.load(output_upha).dataobj)

            assert input_upha.shape  == (312, 384, 256)
            assert output_upha.shape == (312, 384, 256)

            if args.save_phase_combination:
                np.save(f'/home/quahb/caipi_denoising/tmp/{subj_id}_{acceleration}_in_before.npy', in_data)
                np.save(f'/home/quahb/caipi_denoising/tmp/{subj_id}_{acceleration}_out_before.npy', out_data)

            in_data, out_data = np.abs(in_data), np.abs(out_data)
            in_data  = magphase2complex( in_data,  input_upha, rescale=True)
            out_data = magphase2complex(out_data, output_upha, rescale=True)

            if args.save_phase_combination:
                np.save(f'/home/quahb/caipi_denoising/tmp/{subj_id}_{acceleration}_in_after.npy', in_data)
                np.save(f'/home/quahb/caipi_denoising/tmp/{subj_id}_{acceleration}_out_after.npy', out_data)

        # compute metrics
        before_psnr = psnr(in_data,  mask=mask_data, subj_id=subj_id, acceleration=acceleration)
        after_psnr  = psnr(out_data, mask=mask_data, subj_id=subj_id, acceleration=acceleration)

        before_msnr = msnr(in_data,  mask=mask_data, subj_id=subj_id)
        after_msnr  = msnr(out_data, mask=mask_data, subj_id=subj_id)

        before_snr = snr(in_data,  mask=mask_data)
        after_snr  = snr(out_data, mask=mask_data)

        if args.data_type == 'magnitude':
            res_ssim = 0#ssim(in_data, out_data)
        elif args.data_type == 'compleximage':
            res_ssim = 0#ssim(np.abs(in_data), np.abs(out_data))

        res_metrics = [ before_psnr, after_psnr, before_msnr, after_msnr, before_snr, after_snr, res_ssim ]
        res_metrics = [ round(m, PRECISION) for m in res_metrics ]

        res_metrics = {
            'before_psnr': res_metrics[0],
            'after_psnr' : res_metrics[1],
            'before_msnr': res_metrics[2],
            'after_msnr' : res_metrics[3],
            'before_snr' : res_metrics[4],
            'after_snr'  : res_metrics[5],
            'ssim'       : res_metrics[6],
        }

        print(subj_id, acceleration, end=' ')
        for k in res_metrics.keys():
            print(k, res_metrics[k], end=' ')
        print()

        # add placeholders in dictionary
        if acceleration not in res: res[acceleration] = {}
        if subj_id not in res[acceleration]: res[acceleration][subj_id] = []

        # update values in dictionary
        res[acceleration][subj_id] = res_metrics
        reset_plt += 1
        if reset_plt == 3:
            reset_plt = 0
            plt.close()

    #print(res.keys()) # list acceleration
    #print(res[list(res.keys())[0]]) # list subjects

    return

def create_parser():
    parser = argparse.ArgumentParser()
    example_str = 'python run_metrics.py magnitude /path/to/{inputs,outputs}'
    parser.add_argument('--unaccelerated', action='store_true', default=False)
    parser.add_argument('--save_phase_combination', action='store_true', default=False)
    parser.add_argument('--unwrapped_phase_dir', default=None)
    parser.add_argument('file_type', choices=['npy', 'nii'])
    parser.add_argument('data_type', choices=['magnitude', 'mag', 'compleximage', 'complex'])
    parser.add_argument('input_path')
    parser.add_argument('output_path')

    return parser

if __name__ == '__main__':
    main()

import argparse
import nibabel as nib
import numpy as np
import os
import pdb
import pydicom as dicom

from tqdm import tqdm

def main():
    parser = create_parser()
    args = parser.parse_args()

    subj_ids = os.listdir(args.input_path)

    for subj in tqdm(subj_ids, ncols=100):
        modalities = os.listdir( os.path.join(args.input_path, subj) )

        subj_output_folder = os.path.join(args.output_path, subj)
        if not os.path.exists(subj_output_folder):
            os.makedirs(subj_output_folder, exist_ok=True)

        for modality in modalities:
            subj_slice_paths = os.listdir( os.path.join(args.input_path, subj, modality) )
            
            subj_slices = []
            for slc_path in subj_slice_paths:
                sp = os.path.join(args.input_path, subj, modality, slc_path)
                ds = dicom.dcmread(sp)
                subj_slices.append(ds)

            # sort slices
            subj_slices = [
                    slices for slices, _ in sorted(
                        zip(subj_slices, subj_slice_paths),
                        key=lambda pair: pair[0].SliceLocation
                    )
            ]

            if 'pha' in modality:
                subj_slices = [s.pixel_array * s.RescaleSlope + s.RescaleIntercept for s in subj_slices]
            else:
                subj_slices = [s.pixel_array for s in subj_slices]

            subj_data = np.stack(subj_slices, axis=0)
            subj_data = np.moveaxis(subj_data, 0, -1)

            nii_path = os.path.join(args.output_path, subj, modality)
            nii_data = nib.Nifti1Image(subj_data, affine=np.eye(4))
            nib.save(nii_data, f'{nii_path}.nii.gz')

    return

def create_parser():
    parser = argparse.ArgumentParser()
    example_str = 'python dicom2nifti.py /path/to/dicom/folder /path/to/nifti/folder'

    parser.add_argument('input_path')
    parser.add_argument('output_path')

    return parser

if __name__ == '__main__':
    main()

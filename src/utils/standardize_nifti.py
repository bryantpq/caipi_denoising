import nibabel as nib
import numpy as np
import pdb

REFERENCE_NIFTI = '/home/quahb/qsm_analysis/No_CAIPI/CID016/Mag.nii.gz'

def standardize_affine_header(data):
    assert isinstance(data, np.ndarray)
    assert list(data.shape) == [312, 384, 256], f'data.shape: {data.shape}'

    ref_nii = nib.load(REFERENCE_NIFTI)
    nii_data = nib.Nifti1Image(data, affine=ref_nii.affine, header=ref_nii.header)

    if data.dtype != 'float16':
        nii_data.header.set_data_dtype(data.dtype)
    else:
        nii_data.header.set_data_dtype('float32')

    return nii_data

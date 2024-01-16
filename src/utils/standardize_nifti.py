import nibabel as nib
import numpy as np
import pdb

CAVS_NIFTI = '/home/quahb/caipi_denoising/data/niftis/cavsms/{}/{}.nii.gz'
MSREBS_NIFTI = '/home/quahb/caipi_denoising/data/niftis/msrebs_magnitude/{}/EPI_{}.nii.gz'

def standardize_affine_header(data, subj_id, acceleration):
    assert isinstance(data, np.ndarray)
    #assert list(data.shape) == [256, 312, 384], f'Incompatible given data.shape: {data.shape}'

    ref_nii = nib.load(CAVS_NIFTI.format(subj_id, acceleration))
    nii_data = nib.Nifti1Image(data, affine=ref_nii.affine, header=ref_nii.header)

    if data.dtype != 'float16':
        nii_data.header.set_data_dtype(data.dtype)
    else:
        nii_data.header.set_data_dtype('float32')

    return nii_data

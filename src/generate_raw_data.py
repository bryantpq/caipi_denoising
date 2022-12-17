import argparse
import logging
import numpy as np
from tqdm import tqdm
import pdb
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import yaml

from preparation.gen_data import N_SUBJECTS, get_dicoms, get_data_dict
from utils.create_logger import create_logger


def main():
    parser = create_parser()
    args = parser.parse_args()
    create_logger('generate_raw_data', 'info')
    
    # load dicoms to be processed
    data_dict = get_data_dict()
    MAG_MODALITIES = ['3D_T2STAR_segEPI', 'CAIPI1x2', 'CAIPI1x3', 'CAIPI2x2']
    PHA_MODALITIES = ['3D_T2STAR_segEPI_pha', 'CAIPI1x2_pha', 'CAIPI1x3_pha', 'CAIPI2x2_pha']

    mag_data_stack, pha_data_stack = [], []
    mag_names_stack, pha_names_stack = [], []

    logging.info('Loading dicom images...')
    for m, p in zip(MAG_MODALITIES, PHA_MODALITIES):
        # data.shape = [n_subj, 384, 312, 256]
        # paths = [ [subj1], [subj2], ...]
        mag_data, mag_paths = get_dicoms(3, m, data_dict)
        pha_data, pha_paths = get_dicoms(3, p, data_dict)

        subj_ids = [ path[0].split('/')[6] for path in mag_paths ]
        mag_names = [ subj_id + '_' + m for subj_id in subj_ids ]
        pha_names = [ subj_id + '_' + p for subj_id in subj_ids ]

        mag_data_stack.append(mag_data)
        pha_data_stack.append(pha_data)
        mag_names_stack.extend(mag_names)
        pha_names_stack.extend(pha_names)

    mag_data, pha_data = np.vstack(mag_data_stack), np.vstack(pha_data_stack)
    mag_names, pha_names = mag_names_stack, pha_names_stack

    if args.rescale_magnitude:
        logging.info('Rescaling magnitude images to [0, 1]...')
        mag_data = mag_data.astype(np.float16)
        for subj_i in tqdm(range(len(mag_data)), ncols=80, desc='magnitude'):
            temp = mag_data[subj_i]
            r_min, r_max = np.min(temp), np.max(temp)
            t_min, t_max = 0, 1
            num = temp - r_min
            den = r_max - r_min
            temp = (num / den) * (t_max - t_min) + t_min
            mag_data[subj_i] = temp

    if args.rescale_phase:
        logging.info('Rescaling phase images to [-pi, pi]...')
        for subj_i in tqdm(range(len(pha_data)), ncols=80, desc='phase'):
            temp = pha_data[subj_i]
            r_min, r_max = np.min(temp), np.max(temp)
            t_min, t_max = -np.pi, np.pi
            num = temp - r_min
            den = r_max - r_min
            temp = (num / den) * (t_max - t_min) + t_min
            pha_data[subj_i] = temp
    
    # generate raw data
    MAG_PATH = '/home/quahb/caipi_denoising/data/raw_data/magnitude/'
    PHA_PATH = '/home/quahb/caipi_denoising/data/raw_data/phase/'
    COMPLEX_IMAG_PATH = '/home/quahb/caipi_denoising/data/raw_data/complex_image/'
    COMPLEX_FREQ_PATH = '/home/quahb/caipi_denoising/data/raw_data/complex_frequency/'

    logging.info('Saving magnitude images...')
    mag_data = mag_data.astype('float32')
    for data, name in zip(mag_data, mag_names):
        np.save(os.path.join(MAG_PATH, name), data)
    
    logging.info('Saving phase images...')
    pha_data = pha_data.astype('float32')
    for data, name in zip(pha_data, pha_names):
        np.save(os.path.join(PHA_PATH, name), data)

    logging.info('Saving complex data in image and frequency domains...')
    image_shape = mag_data[0].shape
    for mag, pha, name in tqdm(zip(mag_data, pha_data, mag_names), 
                ncols=80, desc='complex data', total=len(mag_names)):
        complex_image = np.zeros(image_shape, dtype=np.csingle)

        complex_image = mag * np.exp(1j * pha)
        complex_image = complex_image.astype('complex64')
        np.save(os.path.join(COMPLEX_IMAG_PATH, name + '_ComplexImage'), complex_image)

        complex_freq = np.fft.fftn(complex_image)
        complex_freq = complex_freq.astype('complex64')
        np.save(os.path.join(COMPLEX_FREQ_PATH, name + '_ComplexFreq'), complex_freq)

    logging.info('Completed generating raw data')

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rescale_magnitude', action='store_true', default=True)
    parser.add_argument('--rescale_phase', action='store_true', default=True)

    return parser

if __name__ == '__main__':
    main()

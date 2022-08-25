import argparse
import logging
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pdb
import tensorflow as tf
from tqdm import tqdm
import yaml

from modeling.get_model import get_model
from preparation.gen_data import get_test_data, get_train_data
from preparation.prepare_tf_dataset import np_to_tfdataset
from preparation.preprocessing_pipeline import preprocess_slices
from utils.data_io import write_slices
from utils.create_logger import create_logger

from patchify import patchify, unpatchify


def main():
    parser = create_parser()
    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    create_logger(config['config_name'], config['logging_level'])

    logging.info(config)
    logging.info('')

    if config['predict_test']['train_or_test_set'] == 'test':
        logging.info('Loading testing data...')
        X_test, slc_paths = get_test_data(config['dimensions'])

        n_volumes = len(X_test) // 256
        test_n_subjects = config['predict_test']['test_n_subjects']

        if type(test_n_subjects) == list:
            keep_indices = test_n_subjects
            test_n_subjects = len(keep_indices)
        else:
            keep_indices = np.random.choice(np.arange(n_volumes), test_n_subjects)

        logging.info('Only keeping {}/{} test volumes for prediction'.format(
                test_n_subjects, len(X_test) // 256 ))
        new_X = []
        new_slc_paths = []
        for i in keep_indices:
            new_X.append(X_test[i*256: i*256 + 256])
            new_slc_paths.extend(slc_paths[i*256: i*256 + 256])
        X_test = np.vstack(new_X)
        slc_paths = new_slc_paths
        logging.info('Subject IDs:')
        logging.info([f.split('/')[6] + '_' + f.split('/')[7]for f in slc_paths[::256] ])

    elif config['predict_test']['train_or_test_set'] == 'train': 
        logging.info('Loading training data...')
        if config['predict_test']['train_leave_one_out']:
            logging.info('Loading 1/5 folds left out from training for prediction...')
            gt_test, X_test, slc_paths = get_train_data(config['dimensions'], train_loo='test')
        else:
            logging.info('Loading 4/5 folds used in training for prediction...')
            gt_test, X_test, slc_paths = get_train_data(config['dimensions'], train_loo='train')
    
    logging.info(f'X_test.shape: {X_test.shape}')
    
    logging.info('Preprocessing X_test')
    X_test = preprocess_slices(X_test,
                               config['dimensions'],
                               config['predict_test']['preprocessing_params'],
                               steps=config['predict_test']['X_steps'])
    logging.info('X_test.shape: {}'.format(X_test.shape))
    
    if 'gt_test' in locals():
        logging.info('Preprocessing gt_test')
        gt_test = preprocess_slices(gt_test,
                                    config['dimensions'],
                                    config['predict_test']['preprocessing_params'],
                                    steps=config['predict_test']['gt_steps'])
        logging.info('gt_test.shape: {}'.format(gt_test.shape))

    logging.info('')
    logging.info('Creating model...')
    strategy = tf.distribute.MirroredStrategy(devices=config['gpus'])
    with strategy.scope():
        model = get_model(model_type=config['model_type'], 
                          input_shape=config['input_shape'],
                          load_model_path=config['predict_test']['load_model_path'])
    
    if config['predict_test']['extract_patches']:
        logging.info('Prediction on patches of slices...')
        patch_size = config['input_shape'][1:3]
        extract_step = config['predict_test']['extract_step']
        
        # TODO
        # Refactor this
        if config['predict_test']['recon_patches']:
            logging.info('    Collecting patches and running prediction...')

            # create all patches
            all_patches = []
            patchify_shape = 0
            for i in tqdm(range(len(X_test)), ncols=80):
                slc = X_test[i][:,:,0]
                patches = patchify(slc, patch_size, step=extract_step)
                patchify_shape = patches.shape[:2]
                patches = patches.reshape(-1, *patch_size)
                patches = np.expand_dims(patches, axis=3)
                all_patches.append(patches)

            # predict on patches
            all_patches = np.vstack(all_patches)
            all_patches = np_to_tfdataset(all_patches)
            patches = model.predict(all_patches,
                                    verbose=1,
                                    batch_size=30)
            patches = patches[:,:,:,0]

            # reconstruct slices from patches
            y_test = []
            n_patches_per_slice = len(patches) // len(X_test)
            for i in range(len(X_test)):
                start_i = i * n_patches_per_slice
                end_i   = i * n_patches_per_slice + n_patches_per_slice
                subj_patches = patches[start_i:end_i]
                subj_patches = subj_patches.reshape(*patchify_shape, *patch_size)
                res_slc = unpatchify(subj_patches, X_test[0][:,:,0].shape)
                y_test.append(res_slc)
            y_test = np.array(y_test)

        # single patch prediction with no recon
        else:
            predict_patch_i = len(patches) // 2
            logging.info(f'Predicting {predict_patch_i}-th patch of the slices. No reconstruction...')
            new_X, y_test = [], []
            for i in range(len(X_test)):
                logging.info( '    Slice {} / {}'.format(i + 1, len(X_test)) )
                slc = X_test[i][:, :, 0]
                patches = patchify(slc, patch_size, step=1)
                patches = patches.reshape(-1, *patch_size)

                patch = patches[predict_patch_i]
                new_X.append(patch)
                patch = np.expand_dims(patch, axis=2)
                patch = np_to_tfdataset(patch)
                res = model.predict(patch,
                                    verbose=1,
                                    batch_size=1)
                res = res[:,:,:,0]
                y_test.append(res)

            X_test = np.array(new_X)
            y_test = np.array(y_test)
    else:
        logging.info('Prediction on full slices ...')
        X_test_tf = np_to_tfdataset(X_test)
        y_test = model.predict(X_test_tf,
                               verbose=1,
                               batch_size=30)

    if X_test.ndim == 4: X_test = X_test[:,:,:,0]

    logging.info('Completed prediction ...')
    logging.info('    Results shape: X_test {}, y_test {}'.format(X_test.shape, y_test.shape))

    logging.info('Saving results...')
    if config['predict_test']['save_mode'] == 'all':
        write_slices(X_test, 'X', config['results_folder'], config['save_dtype'])
        write_slices(y_test, 'y', config['results_folder'], config['save_dtype'])
    elif config['predict_test']['save_mode'] == 'subject':
        n_subjects = len(X_test) // 256
        for i in range(n_subjects):
            cur_X = X_test[i * 256: i * 256 + 256]
            cur_y = y_test[i * 256: i * 256 + 256]
            cur_subj_id = slc_paths[i * 256].split('/')[6]
            cur_modality = slc_paths[i * 256].split('/')[7]

            # clean up images
            cur_X = np.moveaxis(cur_X, 0, 2)
            cur_X = cur_X[:, 36:384 - 36, :]
            cur_y = np.moveaxis(cur_y, 0, 2)
            cur_y = cur_y[:, 36:384 - 36, :]

            write_slices(cur_X, f'{cur_subj_id}_{cur_modality}_X', 
                    config['results_folder'], config['save_dtype'])
            write_slices(cur_y, f'{cur_subj_id}_{cur_modality}_y', 
                    config['results_folder'], config['save_dtype'])

            if 'gt_test' in locals():
                cur_gt = gt_test[i * 256: i * 256 + 256]
                cur_gt = np.moveaxis(cur_gt, 0, 2)
                cur_gt = cur_gt[:, 36:384 - 36, :, 0]

                write_slices(cur_gt, f'{cur_subj_id}_{cur_modality}_gt', 
                        config['results_folder'], config['save_dtype'])

    logging.info('Prediction complete for config: {}'.format(config['config_name']))
    logging.info('Results saved at {}'.format(config['results_folder']))

    
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=argparse.FileType('r'))

    return parser

if __name__ == '__main__':
    main()

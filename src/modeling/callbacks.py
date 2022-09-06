import logging
import os
import tensorflow as tf
from datetime import datetime, date


def get_training_cb(config_name,
                    patience=3, 
                    model_type=None,
                    save_path='/home/quahb/caipi_denoising/models',
                    tensorboard_path='/home/quahb/caipi_denoising/logs/fit'):
    """
    Return list of callbacks to be used in model.fit(cb=cb)
    """
    # CB: Early stopping
    cb_earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    verbose=1,
                                                    patience=patience)
    
    # CB: Checkpoint
    save_path = os.path.join(save_path, config_name + '_' + str(date.today()))
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logging.info("Creating new folder for model weights: {}".format(save_path))
    
    save_filename = 'model{}'.format(model_type) + '_ep{epoch:02d}.h5'
    save_filename = os.path.join(save_path, save_filename)
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_filename,
                                                       verbose=1,
                                                       save_best_only=True,
                                                       monitor='val_loss',
                                                       mode='min')

    # CB: Tensorboard info
    log_dir = os.path.join(tensorboard_path, config_name + '_' + str(date.today()))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        logging.info('Creating new folder for TensorBoard logging: {}'.format(log_dir))
    cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                    histogram_freq=1,
                                                    write_images=False,
                                                    write_graph=False)
    
    cb_list = [cb_earlystop, cb_checkpoint, cb_tensorboard]
    
    return cb_list

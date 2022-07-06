import os
import tensorflow as tf
from datetime import date


def get_training_cb(patience=3, 
                    save_path=None,
                    save_filename=None):
    """
    Return list of callbacks to be used in model.fit(cb=cb)
    """
    
    # CB: Early stopping
    cb_earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                    verbose=1,
                                                    patience=patience)
    
    # CB: Checkpoint
    date_s = str(date.today())
    save_path = save_path + '_' + date_s
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Creating new folder for model weights: {}".format(save_path))
    
    save_filename = os.path.join(save_path, save_filename)
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_filename,
                                                       verbose=1,
                                                       save_best_only=True,
                                                       monitor='loss',
                                                       mode='min')

    log_dir = "/home/quahb/caipi_denoising/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print('Creating new folder for TensorBoard logging: {}'.format(log_dir))
    cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Return CB
    cb_list = [cb_earlystop, cb_checkpoint, cb_tensorboard]
    
    return cb_list
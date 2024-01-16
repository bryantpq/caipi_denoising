import numpy as np
import tensorflow as tf

def complex_split(data):
    real, imag = np.real(data), np.imag(data)
    real, imag = np.squeeze(real), np.squeeze(imag)

    res = np.zeros(arr1.shape[:-1] + (2, ), dtype=np.float32)
    res[:,:,:,0] = real
    res[:,:,:,1] = imag
    
    return res

def np_to_tfdataset(arr1, arr2=None, batch_size=32, trim_batch=False):
    if arr2 is not None:
        assert arr1.shape == arr2.shape, f'Given arrays should have the same shape.'

    if trim_batch:
        limit = arr1.shape[0] // batch_size * batch_size
        arr1 = arr1[:limit]
        if arr2 is not None: arr2 = arr2[:limit]

    if arr2 is not None:
        data = tf.data.Dataset.from_tensor_slices((arr1, arr2))
    else:
        data = tf.data.Dataset.from_tensor_slices(arr1)

    data = data.batch(batch_size)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    
    data = data.with_options(options)

    return data

def merge_datasets(X_train, y_train, X_valid, y_valid, train_size=0.8):
    logging.info('Before merge:')
    logging.info('{}, {}, {}, {}'.format(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape))
    logging.info('After merge:')
    
    images = np.vstack([ X_train, X_valid ])
    labels = np.vstack([ y_train, y_valid ])
    shuffle = np.random.RandomState(seed=42).permutation(len(images))
    images, labels = images[shuffle], labels[shuffle]
    val_i = int( len(images) * train_size )
    X_train, y_train = images[:val_i], labels[:val_i]
    X_valid, y_valid = images[val_i:], labels[val_i:]

    return X_train, y_train, X_valid, y_valid

import numpy as np
import tensorflow as tf

def complex_split(data):
    real, imag = np.real(data), np.imag(data)
    real, imag = np.squeeze(real), np.squeeze(imag)

    res = np.zeros(arr1.shape[:-1] + (2, ), dtype=np.float32)
    res[:,:,:,0] = real
    res[:,:,:,1] = imag
    
    return res

def np_to_tfdataset(arr1, arr2=None, batch_size=32):
    if arr2 is not None:
        data = tf.data.Dataset.from_tensor_slices((arr1, arr2))
    else:
        data = tf.data.Dataset.from_tensor_slices(arr1)

    data = data.batch(batch_size)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    
    data = data.with_options(options)

    return data

import numpy as np
import tensorflow as tf

def np_to_tfdataset(arr1, arr2=None, batch_size=32, complex_split=False):
    if complex_split:
        real, imag = np.real(arr1), np.imag(arr1)
        real, imag = np.squeeze(real), np.squeeze(imag)
        res = np.zeros(arr1.shape[:-1] + (2, ), dtype=np.float16)
        res[:,:,:,0] = real
        res[:,:,:,1] = imag
        arr1 = res

        if arr2 is not None:
            real, imag = np.real(arr2), np.imag(arr2)
            real, imag = np.squeeze(real), np.squeeze(imag)
            res = np.zeros(arr2.shape[:-1] + (2, ), dtype=np.float16)
            res[:,:,:,0] = real
            res[:,:,:,1] = imag
            arr2 = res

    if arr2 is not None:
        data = tf.data.Dataset.from_tensor_slices((arr1, arr2))
    else:
        data = tf.data.Dataset.from_tensor_slices(arr1)

    data = data.batch(batch_size)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    
    data = data.with_options(options)

    return data

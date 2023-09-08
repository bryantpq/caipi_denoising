import tensorflow as tf

from scipy.fftpack import dct, dst, fft, ifft, idct
from tensorflow.keras.layers import Layer

def dct2(a, axes=[1, 2]):
    return dct( dct( a, axis=axes[0]), axis=axes[1])

def idct2(a, axes=[1, 2]):
    return idct( idct( a, axis=axes[0]), axis=axes[1])

class DCTConvLayer(Layer):
    def __init__(
            self, 
            window_size=7, 
            dtype=tf.dtypes.float32,
            **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.my_dtype = dtype

    def build(self, batch_input_shape):
        self.batch_input_shape = batch_input_shape
        super().build(batch_input_shape)

    def get_config(self):
        base_config = super().get_config()

        return {**base_config}

    @tf.function
    def call(self, X):
        # X = [None, 384, 384, 1]
        input_shape = self.batch_input_shape
        window_size = self.window_size

        hf_stack_shape = tf.concat( [tf.shape(X)[:-1], tf.constant([48])], 0)
        zf_stack = tf.zeros([1228, 384, 384, 1], dtype=self.my_dtype, name='zf_stack') # [None, 384, 384, 1]
        hf_stack = tf.zeros([1228, 384, 384, 48], dtype=self.my_dtype, name='hf_stack') # [None, 384, 384, 48]

        tf.print(zf_stack.shape, hf_stack.shape)
        tf.print(tf.shape(zf_stack), tf.shape(hf_stack))

        for i in range(input_shape[1] - window_size):
            for j in range(input_shape[2] - window_size):
                patch = X[:, i:i + window_size, j:j + window_size, :] # [None, 7, 7, 1]
                dct_patch = tf.py_function(func=dct2, inp=[patch], Tout=tf.float32) # [None, 7, 7, 1]
                vect_shape = [tf.shape(X)[0], window_size * window_size]
                dct_vect = tf.reshape(dct_patch, vect_shape) # [None, 49]
                zf_img, hf_img = dct_vect[:, 0], dct_vect[:, 1:]

                #tf.scatter_nd(
                #        indices=tf.constant([ [k, i, j] for k in range(tf.shape(X)[0]) ]),
                #        updates=zf_img,
                #        shape=zf_stack
                #)
                zf_stack = zf_stack.assign(zf_img)
                hf_stack = hf_stack.assign(hf_img)

        return zf_stack, hf_stack

class IDCTConvLayer(Layer):
    def __init__(self, im_zero_freq, window_size=7, dtype=tf.dtypes.float32):
        super(IDCTConv, self).__init__()
        self.im_zero_freq = im_zero_freq
        self.window_size = window_size
        self.my_dtype = dtype

    def build(self, input_shape):
        self.my_input_shape = input_shape

    def call(self, X):
        # [None, 384, 384, 48]
        input_shape = self.my_input_shape
        window_size = self.window_size
        im_zero_freq = self.im_zero_freq
        im_high_freq = X

        denoised_image = tf.zeros(X.shape[:-1] + (1, ), # [None, 384, 384, 1]
                dtype=self.my_dtype, name='denoised_image')

        for i in range(input_shape[1] - window_size):
            for j in range(input_shape[2] - window_size):
                patch = tf.zeros(X.shape[0] + (window_size * window_size, 1), # [None, 49, 1]
                        dtype=self.my_dtype)
                patch[:, 0, :] = im_zero_freq[:, i, j, :] # [None, 1, 1] = [None, 1, 1, 1]
                patch[:, 1:, :] = im_high_freq[:, i, j, :]
                patch = patch.reshape(patch.shape[0] + (window_size, window_size, 1)) # [None, 7, 7, 1]
                idct_patch = idct2(patch, axes=[1, 2]) # [None, 7, 7, 1]
                denoised_image[:, i:i + window_size, j:j + window_size, :] = idct_patch
        
        return denoised_image

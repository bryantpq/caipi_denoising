import numpy as np
import sys
import tensorflow as tf
import pdb

from math import pi
from scipy.fftpack import dct, idct
from tensorflow.keras.layers import Layer


def complex_batch_norm(
    tf_input,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    **kwargs
):
    pass

def complex_conv2d(
    tf_input, num_features, kernel_size, stride=1, data_format="channels_last", 
    dilation_rate=(1, 1), use_bias=True, kernel_initializer=None, kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, 
    bias_constraint=None, trainable=True, activation=None
):
    # allocate half the features to real, half to imaginary
    num_features = num_features // 2

    tf_real = tf.math.real(tf_input)
    tf_imag = tf.math.imag(tf_input)

    tf_real_real = tf.keras.layers.Conv2D(
        filters=num_features,
        kernel_size=kernel_size,
        strides=[stride, stride],
        padding="same",
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
#        name="real_conv",
    )(tf_real)
    tf_imag_real = tf.keras.layers.Conv2D(
        filters=num_features,
        kernel_size=kernel_size,
        strides=[stride, stride],
        padding="same",
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
#        name="real_conv",
    )(tf_imag)
    tf_real_imag = tf.keras.layers.Conv2D(
        filters=num_features,
        kernel_size=kernel_size,
        strides=[stride, stride],
        padding="same",
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
#        name="imag_conv",
    )(tf_real)
    tf_imag_imag = tf.keras.layers.Conv2D(
        filters=num_features,
        kernel_size=kernel_size,
        strides=[stride, stride],
        padding="same",
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
#        name="imag_conv",
    )(tf_imag)
    real_out = tf_real_real - tf_imag_imag
    imag_out = tf_imag_real + tf_real_imag
    tf_output = tf.dtypes.complex(real_out, imag_out)

    return tf_output


def complex_conv3d(
    tf_input, num_features, kernel_size, stride=1, data_format="channels_last", 
    dilation_rate=(1, 1, 1), use_bias=True, kernel_initializer=None, kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, 
    bias_constraint=None, trainable=True, activation=None
):
    # allocate half the features to real, half to imaginary
    num_features = num_features // 2

    tf_real = tf.math.real(tf_input)
    tf_imag = tf.math.imag(tf_input)

    tf_real_real = tf.keras.layers.Conv3D(
        filters=num_features,
        kernel_size=kernel_size,
        strides=[stride, stride, stride],
        padding="same",
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
#        name="real_conv",
    )(tf_real)
    tf_imag_real = tf.keras.layers.Conv3D(
        filters=num_features,
        kernel_size=kernel_size,
        strides=[stride, stride, stride],
        padding="same",
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
#        name="real_conv",
    )(tf_imag)
    tf_real_imag = tf.keras.layers.Conv3D(
        filters=num_features,
        kernel_size=kernel_size,
        strides=[stride, stride, stride],
        padding="same",
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
#        name="imag_conv",
    )(tf_real)
    tf_imag_imag = tf.keras.layers.Conv3D(
        filters=num_features,
        kernel_size=kernel_size,
        strides=[stride, stride, stride],
        padding="same",
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
#        name="imag_conv",
    )(tf_imag)
    real_out = tf_real_real - tf_imag_imag
    imag_out = tf_imag_real + tf_real_imag
    tf_output = tf.dtypes.complex(real_out, imag_out)

    return tf_output


def complex_conv_transpose(tf_input, num_features, kernel_size, stride, data_format="channels_last", use_bias=True,
                           kernel_initializer=None, kernel_regularizer=None, bias_regularizer=None,
                           activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True
                           ):
    # allocate half the features to real, half to imaginary
    # num_features = num_features // 2

    tf_real = tf.math.real(tf_input)
    tf_imag = tf.math.imag(tf_input)

    tf_real_real = tf.keras.layers.Conv2DTranspose(
        filters=num_features,
        kernel_size=kernel_size,
        strides=[stride, stride],
        padding="same",
        data_format=data_format,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
#        name="real_conv",
    )(tf_real)
    tf_imag_real = tf.keras.layers.Conv2DTranspose(
        filters=num_features,
        kernel_size=kernel_size,
        strides=[stride, stride],
        padding="same",
        data_format=data_format,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
#        name="real_conv",
    )(tf_imag)
    tf_real_imag = tf.keras.layers.Conv2DTranspose(
        filters=num_features,
        kernel_size=kernel_size,
        strides=[stride, stride],
        padding="same",
        data_format=data_format,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
#        name="imag_conv",
    )(tf_real)
    tf_imag_imag = tf.keras.layers.Conv2DTranspose(
        filters=num_features,
        kernel_size=kernel_size,
        strides=[stride, stride],
        padding="same",
        data_format=data_format,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
#        name="imag_conv",
    )(tf_imag)
    real_out = tf_real_real - tf_imag_imag
    imag_out = tf_imag_real + tf_real_imag
    tf_output = tf.dtypes.complex(real_out, imag_out)

    return tf_output


def complex_conv1d(
    tf_input, num_features, kernel_size, stride=1, data_format="channels_last", dilation_rate=(1), use_bias=True,
    kernel_initializer=None, kernel_regularizer=None, bias_regularizer=None,
    activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True
):
    # allocate half the features to real, half to imaginary
    num_features = num_features // 2

    tf_real = tf.math.real(tf_input)
    tf_imag = tf.math.imag(tf_input)

    tf_real_real = tf.keras.layers.Conv1D(
        inputs=tf_real,
        filters=num_features,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
#        name="real_conv",
    )
    tf_imag_real = tf.keras.layers.Conv1D(
        tf_imag,
        filters=num_features,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
#        name="real_conv",
    )
    tf_real_imag = tf.keras.layers.Conv1D(
        tf_real,
        filters=num_features,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
#        name="imag_conv",
    )
    tf_imag_imag = tf.keras.layers.Conv1D(
        tf_imag,
        filters=num_features,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
#        name="imag_conv",
    )
    real_out = tf_real_real - tf_imag_imag
    imag_out = tf_imag_real + tf_real_imag
    tf_output = tf.dtypes.complex(real_out, imag_out)

    return tf_output


def crelu(x):
    real_x, imag_x = tf.math.real(x), tf.math.imag(x)
    out_real_x = tf.keras.layers.Activation('relu')(real_x)
    out_imag_x = tf.keras.layers.Activation('relu')(imag_x)

    output = tf.dtypes.complex(out_real_x, out_imag_x)

    return output


def zrelu(x):
    # x and tf_output are complex-valued
    phase = tf.math.angle(x)

    # Check whether phase <= pi/2
    le = tf.math.less_equal(phase, pi / 2)

    # if phase <= pi/2, keep it in comp
    # if phase > pi/2, throw it away and set comp equal to 0
    y = tf.zeros_like(x)
    x = tf.where(le, x, y)

    # Check whether phase >= 0
    ge = tf.math.greater_equal(phase, 0)

    # if phase >= 0, keep it
    # if phase < 0, throw it away and set output equal to 0
    output = tf.where(ge, x, y)

    return output


def modrelu(x, data_format="channels_last"):
    input_shape = tf.shape(x)
    if data_format == "channels_last":
        axis_z = 1
        axis_y = 2
        axis_c = 3
    else:
        axis_c = 1
        axis_z = 2
        axis_y = 3

    # Channel size
    shape_c = x.shape[axis_c]

    with tf.name_scope("bias") as scope:
        if data_format == "channels_last":
            bias_shape = (1, 1, 1, shape_c)
        else:
            bias_shape = (1, shape_c, 1, 1)
        bias = tf.get_variable(name=scope,
                               shape=bias_shape,
                               initializer=tf.constant_initializer(0.0),
                               trainable=True)
    # relu(|z|+b) * (z / |z|)
    norm = tf.math.abs(x)
    scale = tf.nn.relu(norm + bias) / (norm + 1e-6)
    output = tf.dtypes.complex(tf.math.real(x) * scale,
                             tf.math.imag(x) * scale)

    return output


def cardioid(x):
    phase = tf.math.angle(x)
    scale = 0.5 * (1 + tf.math.cos(phase))
    output = tf.complex(tf.math.real(x) * scale, tf.math.imag(x) * scale)
    # output = 0.5*(1+tf.cos(phase))*z

    return output


def _softshrink(x, lower=-0.5, upper=0.5):
    '''
    Taken from https://github.com/tensorflow/addons/blob/v0.17.0/tensorflow_addons/activations/softshrink.py#L21
    '''
    #if lower > upper:
    #    raise ValueError(
    #        "The value of lower is {} and should"
    #        " not be higher than the value "
    #        "variable upper, which is {} .".format(lower, upper)
    #    )
    x = tf.convert_to_tensor(x)
    values_below_lower = tf.where(x < lower, x - lower, 0)
    values_above_upper = tf.where(upper < x, x - upper, 0)

    return values_below_lower + values_above_upper

def softshrink(x):
    alpha = tf.Variable(0.001, dtype='float32', trainable=True)
    sigma = tf.math.reduce_std(x)

    lower_th, upper_th = -1 * alpha * sigma, alpha * sigma

    z = _softshrink(x, lower_th, upper_th)

    return z


class SSLayer(Layer):
    def __init__(
            self, 
            noise_window_size=[32, 32],
            init_alpha=[0.0001, 0.01],
            relu_mode=False,
            **kwargs):
        super().__init__(**kwargs)
        self.noise_window_size = noise_window_size
        self.init_alpha = init_alpha
        self.relu_mode = relu_mode

    def build(self, batch_input_shape):
        initializer = tf.keras.initializers.RandomUniform(
                minval=self.init_alpha[0], maxval=self.init_alpha[1])
        self.count = 0

        if not self.relu_mode:
            self.alpha = self.add_weight(
                    name='alpha',
                    shape=[1],
                    dtype=tf.dtypes.float32,
                    trainable=True,
                    initializer=initializer)

        super().build(batch_input_shape)

    def call(self, X):
        batch_mid_slice = int(tf.shape(X)[0] / 2)
        img_mid_idx = 384 #int(tf.shape(X)[1] / 2)
        start, end = img_mid_idx - self.noise_window_size[0], img_mid_idx + self.noise_window_size[1]
        
        std = tf.math.reduce_std(X[batch_mid_slice, start:end, start:end, :] - \
                X[batch_mid_slice - 1, start:end, start:end, :])

        self.sigma = std 
        #tf.print(self.count, ': ', std, end=', ')

        self.count += 1
        #if not tf.math.is_nan(std): # only keep sigma if its not NaN
        #    self.sigma = std 
        #else:
        #    tf.print('Ooops!')
        #    self.sigma = tf.random.uniform(shape=[], minval=0.3, maxval=0.8, dtype=tf.float32)

        if self.relu_mode:
            lower_th = -9999999.0
            upper_th = 0.0
        else:
            lower_th = self.alpha * self.sigma * -1
            upper_th = self.alpha * self.sigma
            #lower_th, upper_th = self.alpha * -1, self.alpha

        return _softshrink(X, lower_th, upper_th)

    def get_config(self):
        base_config = super().get_config()

        return {**base_config, 
                'init_alpha': self.init_alpha,
                'relu_mode': self.relu_mode}

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

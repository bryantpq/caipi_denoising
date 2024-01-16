import numpy as np
import tensorflow as tf
import pdb

from math import pi


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

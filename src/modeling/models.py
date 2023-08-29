import pdb
import sys
import tensorflow as tf

from tensorflow.keras import layers, losses, Sequential, Input
from tensorflow.keras.models import Model

from modeling.custom_utils import DCTConvLayer, IDCTConvLayer, SSLayer
from modeling.custom_utils import complex_conv2d, complex_conv3d, complex_conv_transpose, complex_conv1d
from modeling.custom_utils import crelu, zrelu, modrelu, cardioid, softshrink


def dncnn(
        dimensions,
        input_shape, 
        n_features=64, 
        n_hidden_layers=15, 
        kernel_size=3,
        residual_layer=False
    ):
    
    def conv_layer(dimensions, n_features, activation=None):
        if dimensions == 2:
            return layers.Conv2D(
                    n_features, 
                    (kernel_size, kernel_size), 
                    activation=activation,
                    padding='same', 
                    strides=1
            )
        elif dimensions == 3:
            return layers.Conv3D(
                    n_features, 
                    (kernel_size, kernel_size, kernel_size), 
                    activation=activation,
                    padding='same', 
                    strides=1
            )

    in_layer = Input(shape=input_shape, dtype=tf.float16)

    block = conv_layer(dimensions, n_features)(in_layer)
    block = layers.Activation('relu')(block)

    for i in range(n_hidden_layers):
        block = conv_layer(dimensions, n_features)(block)
        block = layers.BatchNormalization()(block)
        block = layers.Activation('relu')(block)

    output = conv_layer(dimensions, input_shape[-1], activation='sigmoid')(block)
    
    if residual_layer: output = layers.Add()([in_layer, output])

    return Model(inputs=[in_layer], outputs=[output])

def cdncnn(
        dimensions,
        input_shape, 
        n_features=128, 
        n_hidden_layers=15,
        kernel_size=3,
        residual_layer=False
    ):

    def conv_layer(dimensions, tf_input, n_features, kernel_size):
        if dimensions == 2:
            return complex_conv2d(tf_input, n_features, kernel_size)
        elif dimensions == 3:
            return complex_conv3d(tf_input, n_features, kernel_size)

    in_layer = Input(shape=input_shape, dtype=tf.complex64)

    block = conv_layer(dimensions, in_layer, n_features, kernel_size)
    block = crelu(block)

    for i in range(n_hidden_layers):
        block = conv_layer(dimensions, block, n_features, kernel_size)
        block = crelu(block)

    N_FEATURES_OUT = 2 # 1 for real, 1 for imag
    output = conv_layer(dimensions, block, N_FEATURES_OUT, kernel_size)

    if residual_layer: output = layers.Add()([in_layer, output])

    return Model(inputs=[in_layer], outputs=[output])

def scnn(
        input_shape, 
        n_features=64, 
        n_hidden_layers=15, 
        kernel_size=3,
        residual_layer=False,
        relu_mode=False,
        init_alpha=None,
        noise_window_size=[32, 32]
    ):
    in_layer = Input(shape=input_shape, dtype=tf.float32)

    block = layers.Conv2D(n_features, (kernel_size, kernel_size), padding='same', strides=1)(in_layer)
    block = SSLayer(noise_window_size=noise_window_size, init_alpha=init_alpha, relu_mode=relu_mode)(block)

    for i in range(n_hidden_layers):
        block = layers.Conv2D(n_features, (kernel_size, kernel_size), padding='same', strides=1)(block)
        block = layers.BatchNormalization()(block)
        block = SSLayer(noise_window_size=noise_window_size, init_alpha=init_alpha, relu_mode=relu_mode)(block)

    output = layers.Conv2D(input_shape[-1], kernel_size=(kernel_size, kernel_size), activation='sigmoid', padding='same')(block)
    
    if residual_layer: output = layers.Add()([in_layer, output])

    return Model(inputs=[in_layer], outputs=[output])

def ddlr(
        input_shape,
        n_features=48,
        window_size=7,
        n_hidden_layers=22,
        kernel_size=3
    ):

    # Input layer converts input_shape from [384, 384, 1] -> [None, 384, 384, 1], basically adds None in front
    in_layer = Input(shape=input_shape, dtype=tf.float32)

    zero_freq, block = DCTConvLayer(window_size=window_size, dtype=tf.float32)(in_layer)
    block = layers.Activation('relu')(block)

    for i in range(n_hidden_layers):
        block = layers.Conv2D(n_features, (kernel_size, kernel_size), padding='same', strides=1)(block)
        block = layers.Activation('relu')(block)

    output = IDCTConvLayer(zero_freq, window_size=window_size, dtype=tf.float32)(block)
    
    return Model(inputs=[in_layer], outputs=[output])

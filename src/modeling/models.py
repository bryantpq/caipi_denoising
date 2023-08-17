import sys
import tensorflow as tf

from tensorflow.keras import layers, losses, Sequential, Input
from tensorflow.keras.models import Model

from modeling.custom_utils import DCTConvLayer, IDCTConvLayer, SSLayer
from modeling.custom_utils import complex_conv, complex_conv_transpose, complex_conv1d
from modeling.custom_utils import crelu, zrelu, modrelu, cardioid, softshrink


def dncnn(
        input_shape, 
        n_features=64, 
        n_hidden_layers=15, 
        kernel_size=3,
        residual_layer=False
    ):
    in_layer  = Input(shape=input_shape, dtype=tf.float32)

    block = layers.Conv2D(n_features, (kernel_size, kernel_size), padding='same', strides=1)(in_layer)
    block = layers.Activation('relu')(block)

    for i in range(n_hidden_layers):
        block = layers.Conv2D(n_features, (kernel_size, kernel_size), padding='same', strides=1)(block)
        block = layers.BatchNormalization()(block)
        block = layers.Activation('relu')(block)

    output = layers.Conv2D(input_shape[-1], kernel_size=(kernel_size, kernel_size), activation='sigmoid', padding='same')(block)
    
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

def complex_dncnn(
        input_shape, 
        n_features=128, 
        n_hidden_layers=15,
        kernel_size=3,
        residual_layer=True
    ):
    in_layer = Input(shape=input_shape, dtype=tf.complex64)

    block = complex_conv(in_layer, n_features, kernel_size)
    block = crelu(block)

    for i in range(n_hidden_layers):
        block = complex_conv(block, n_features, kernel_size)
        block = crelu(block)

    n_features_out = 2 # use 2 instead of 1 because the layer uses half for real and half for imag
    output = complex_conv(block, n_features_out, kernel_size)

    if residual_layer:
        return Model(inputs=[in_layer], outputs=[in_layer - output])
    else:
        return Model(inputs=[in_layer], outputs=[output])

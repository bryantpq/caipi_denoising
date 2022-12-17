import tensorflow as tf

from tensorflow.keras import layers, losses, Sequential, Input
from tensorflow.keras.models import Model

from modeling.complex_utils import complex_conv, complex_conv_transpose, complex_conv1d
from modeling.complex_utils import crelu, zrelu, modrelu, cardioid


def dncnn(
        input_shape, 
        n_features=64, 
        n_hidden_layers=15, 
        kernel_size=3
    ):
    in_layer  = Input(shape=input_shape, dtype=tf.float32)

    block = layers.Conv2D(n_features, (kernel_size, kernel_size), padding='same', strides=1)(in_layer)
    block = layers.Activation('relu')(block)

    for i in range(n_hidden_layers):
        block = layers.Conv2D(n_features, (kernel_size, kernel_size), padding='same', strides=1)(block)
        block = layers.BatchNormalization()(block)
        block = layers.Activation('relu')(block)

    output = layers.Conv2D(input_shape[-1], kernel_size=(kernel_size, kernel_size), activation='sigmoid', padding='same')(block)
    
    return Model(inputs=[in_layer], outputs=[output])

def res_dncnn(input_shape):
    n_layers = 15
    in_layer = Input(shape=input_shape, dtype=tf.float32)

    block = layers.Conv2D(64, (3, 3), padding='same', strides=1)(in_layer)
    block = layers.Activation('relu')(block)

    for i in range(n_layers):
        block = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block)
        block = layers.BatchNormalization()(block)
        block = layers.Activation('relu')(block)

    output = layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(block)
    #output = layers.Subtract()([in_layer, output])
    
    return Model(inputs=[in_layer], outputs=[in_layer - output])

def complex_dncnn(
        input_shape, 
        n_features=128, 
        n_hidden_layers=10, 
        kernel_size=3,
        residual_layer=False
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
        return Model(inputs=[in_layer], outputs=[in_layer + output])
    else:
        return Model(inputs=[in_layer], outputs=[output])

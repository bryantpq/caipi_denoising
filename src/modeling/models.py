import tensorflow as tf

from tensorflow.keras import layers, losses, Sequential, Input
from tensorflow.keras.models import Model


class Denoiser(Model):
    def __init__(self, input_shape):
        super(Denoiser, self).__init__()

        self.model = Sequential([
            
            layers.InputLayer(input_shape),
            layers.Conv2D(256, (3, 3), padding='same', strides=2),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            layers.Conv2D(128, (3, 3), padding='same', strides=2),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            layers.Conv2D(64,  (3, 3), padding='same', strides=2),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            layers.Conv2D(32,  (3, 3), padding='same', strides=2),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            
            layers.Conv2DTranspose(32,  kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            layers.Conv2DTranspose(64,  kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'),
        ])

    def call(self, x):
        result = self.model(x)
        
        return result
    
    
def get_model1(input_shape):
    enc_input  = Input(shape=input_shape, dtype=tf.float32)

    enc_block1 = layers.Conv2D(256, (3, 3), padding='same', strides=1)(enc_input)
    enc_block1 = layers.BatchNormalization()(enc_block1)
    enc_block1 = layers.Activation('relu')(enc_block1)
    enc_block1 = layers.MaxPool2D(pool_size=(2, 2))(enc_block1)

    enc_block2 = layers.Conv2D(128, (3, 3), padding='same', strides=1)(enc_block1)
    enc_block2 = layers.BatchNormalization()(enc_block2)
    enc_block2 = layers.Activation('relu')(enc_block2)
    enc_block2 = layers.MaxPool2D(pool_size=(2, 2))(enc_block2)

    enc_block3 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(enc_block2)
    enc_block3 = layers.BatchNormalization()(enc_block3)
    enc_block3 = layers.Activation('relu')(enc_block3)
    enc_block3 = layers.MaxPool2D(pool_size=(2, 2))(enc_block3)

    enc_block4 = layers.Conv2D(32, (3, 3), padding='same', strides=1)(enc_block3)
    enc_block4 = layers.BatchNormalization()(enc_block4)
    enc_block4 = layers.Activation('relu')(enc_block4)
    enc_block4 = layers.MaxPool2D(pool_size=(2, 2))(enc_block4)

    # End encoder
    # Encoder output/Decoder input: (24, 24, 32)
    dec_block1 = layers.Conv2DTranspose(64,  kernel_size=3, strides=2, padding='same')(enc_block4)
    dec_block1 = layers.BatchNormalization()(dec_block1)
    dec_block1 = layers.Activation('relu')(dec_block1)

    concat1 = layers.Concatenate()([dec_block1, enc_block3])
    
    dec_block2 = layers.Conv2DTranspose(128,  kernel_size=3, strides=2, padding='same')(concat1)
    dec_block2 = layers.BatchNormalization()(dec_block2)
    dec_block2 = layers.Activation('relu')(dec_block2)

    concat2 = layers.Concatenate()([dec_block2, enc_block2])

    dec_block3 = layers.Conv2DTranspose(256,  kernel_size=3, strides=2, padding='same')(concat2)
    dec_block3 = layers.BatchNormalization()(dec_block3)
    dec_block3 = layers.Activation('relu')(dec_block3)

    dec_block4 = layers.Conv2DTranspose(256,  kernel_size=3, strides=2, padding='same')(dec_block3)
    dec_block4 = layers.BatchNormalization()(dec_block4)
    dec_block4 = layers.Activation('relu')(dec_block4)


    dec_output = layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(dec_block4)

    
    return Model(inputs=[enc_input], outputs=[dec_output])


def get_model2(input_shape):
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


def get_model3(input_shape, n_hidden_layers=15, kernel_size=3):
    in_layer  = Input(shape=input_shape, dtype=tf.float32)

    block = layers.Conv2D(64, (kernel_size, kernel_size), padding='same', strides=1)(in_layer)
    block = layers.Activation('relu')(block)

    for i in range(n_hidden_layers):
        block = layers.Conv2D(64, (kernel_size, kernel_size), padding='same', strides=1)(block)
        block = layers.BatchNormalization()(block)
        block = layers.Activation('relu')(block)

    output = layers.Conv2D(1, kernel_size=(kernel_size, kernel_size), activation='sigmoid', padding='same')(block)
    
    return Model(inputs=[in_layer], outputs=[output])

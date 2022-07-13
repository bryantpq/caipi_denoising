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
    input  = Input(shape=input_shape, dtype=tf.float32)

    block0 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(input)
    block0 = layers.Activation('relu')(block0)

    block1 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block0)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Activation('relu')(block1)

    block2 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block1)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Activation('relu')(block2)

    block3 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block2)
    block3 = layers.BatchNormalization()(block3)
    block3 = layers.Activation('relu')(block3)

    block4 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block3)
    block4 = layers.BatchNormalization()(block4)
    block4 = layers.Activation('relu')(block4)

    block5 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block4)
    block5 = layers.BatchNormalization()(block5)
    block5 = layers.Activation('relu')(block5)

    block6 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block5)
    block6 = layers.BatchNormalization()(block6)
    block6 = layers.Activation('relu')(block6)

    block7 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block6)
    block7 = layers.BatchNormalization()(block7)
    block7 = layers.Activation('relu')(block7)

    block8 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block7)
    block8 = layers.BatchNormalization()(block8)
    block8 = layers.Activation('relu')(block8)

    output = layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(block8)

    return Model(inputs=[input], outputs=[output])


def get_model3(input_shape):
    input  = Input(shape=input_shape, dtype=tf.float32)

    block0 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(input)
    block0 = layers.Activation('relu')(block0)

    block1 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block0)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Activation('relu')(block1)

    block2 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block1)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Activation('relu')(block2)

    block3 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block2)
    block3 = layers.BatchNormalization()(block3)
    block3 = layers.Activation('relu')(block3)

    block4 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block3)
    block4 = layers.BatchNormalization()(block4)
    block4 = layers.Activation('relu')(block4)

    block5 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block4)
    block5 = layers.BatchNormalization()(block5)
    block5 = layers.Activation('relu')(block5)

    block6 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block5)
    block6 = layers.BatchNormalization()(block6)
    block6 = layers.Activation('relu')(block6)

    block7 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block6)
    block7 = layers.BatchNormalization()(block7)
    block7 = layers.Activation('relu')(block7)

    block8 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block7)
    block8 = layers.BatchNormalization()(block8)
    block8 = layers.Activation('relu')(block8)

    block9 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block8)
    block9 = layers.BatchNormalization()(block9)
    block9 = layers.Activation('relu')(block9)

    block10 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block9)
    block10 = layers.BatchNormalization()(block10)
    block10 = layers.Activation('relu')(block10)

    block11 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block10)
    block11 = layers.BatchNormalization()(block11)
    block11 = layers.Activation('relu')(block11)

    block12 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block11)
    block12 = layers.BatchNormalization()(block12)
    block12 = layers.Activation('relu')(block12)

    block13 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block12)
    block13 = layers.BatchNormalization()(block13)
    block13 = layers.Activation('relu')(block13)

    block14 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block13)
    block14 = layers.BatchNormalization()(block14)
    block14 = layers.Activation('relu')(block14)

    block15 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block14)
    block15 = layers.BatchNormalization()(block15)
    block15 = layers.Activation('relu')(block15)

    output = layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(block15)
    
    return Model(inputs=[input], outputs=[output])


def get_model4(input_shape):
    input  = Input(shape=input_shape, dtype=tf.float32)

    block0 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(input)
    block0 = layers.Activation('relu')(block0)

    block1 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block0)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Activation('relu')(block1)

    block2 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block1)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Activation('relu')(block2)

    block3 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block2)
    block3 = layers.BatchNormalization()(block3)
    block3 = layers.Activation('relu')(block3)

    block4 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block3)
    block4 = layers.BatchNormalization()(block4)
    block4 = layers.Activation('relu')(block4)

    block5 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block4)
    block5 = layers.BatchNormalization()(block5)
    block5 = layers.Activation('relu')(block5)

    block6 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block5)
    block6 = layers.BatchNormalization()(block6)
    block6 = layers.Activation('relu')(block6)

    block7 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block6)
    block7 = layers.BatchNormalization()(block7)
    block7 = layers.Activation('relu')(block7)

    block8 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block7)
    block8 = layers.BatchNormalization()(block8)
    block8 = layers.Activation('relu')(block8)

    block9 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block8)
    block9 = layers.BatchNormalization()(block9)
    block9 = layers.Activation('relu')(block9)

    block10 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block9)
    block10 = layers.BatchNormalization()(block10)
    block10 = layers.Activation('relu')(block10)

    block11 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block10)
    block11 = layers.BatchNormalization()(block11)
    block11 = layers.Activation('relu')(block11)

    block12 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block11)
    block12 = layers.BatchNormalization()(block12)
    block12 = layers.Activation('relu')(block12)

    block13 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block12)
    block13 = layers.BatchNormalization()(block13)
    block13 = layers.Activation('relu')(block13)

    block14 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block13)
    block14 = layers.BatchNormalization()(block14)
    block14 = layers.Activation('relu')(block14)

    block15 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block14)
    block15 = layers.BatchNormalization()(block15)
    block15 = layers.Activation('relu')(block15)

    block16 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block15)
    block16 = layers.BatchNormalization()(block16)
    block16 = layers.Activation('relu')(block16)

    block17 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block16)
    block17 = layers.BatchNormalization()(block17)
    block17 = layers.Activation('relu')(block17)

    block18 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block17)
    block18 = layers.BatchNormalization()(block18)
    block18 = layers.Activation('relu')(block18)

    block19 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block18)
    block19 = layers.BatchNormalization()(block19)
    block19 = layers.Activation('relu')(block19)

    block20 = layers.Conv2D(64, (3, 3), padding='same', strides=1)(block19)
    block20 = layers.BatchNormalization()(block20)
    block20 = layers.Activation('relu')(block20)

    output = layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(block20)

    return Model(inputs=[input], outputs=[output])

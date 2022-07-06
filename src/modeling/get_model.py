import os
import tensorflow as tf

from tensorflow.keras import layers, losses, Sequential, Input
from tensorflow.keras.models import Model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_model(model_type, input_shape=None, load_model_path=None):
    assert len(input_shape) == 4, "Expected input_shape to be 4-dim. Got {}".format(input_shape)
    
    img_size = input_shape[1:]
    
    if model_type == 0:
        print('    Using model type 0')
        model = Denoiser(input_shape=img_size)
        
    elif model_type == 1:
        print('    Using model type 1')
        model = get_denoiser(input_shape=img_size)

    model.build(input_shape=input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=losses.MeanSquaredError(), 
                  metrics=[])
    
    if load_model_path is not None:
        print('    Loading model weights: {}'.format(load_model_path))
        model.load_weights(load_model_path)
    
    return model


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
    
    
def get_denoiser(input_shape):
    enc_input  = Input(shape=input_shape, dtype=tf.float32)

    enc_block1 = layers.Conv2D(256, (3, 3), padding='same', strides=1, name='test')(enc_input)
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
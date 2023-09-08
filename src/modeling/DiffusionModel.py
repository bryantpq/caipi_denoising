import math
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from numpy import r_
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

from modeling.complex_utils import complex_conv2d, crelu
from utils.dct import dct2, idct2
from utils.vizualization_tools import plot2, plot4, plot_slices, plot_patches

# Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj


class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[3]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)

        temb = activation_fn(t)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[
            :, None, None, :
        ]

        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)

        x = layers.Add()([x, temb])
        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
        )(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownSample(width):
    def apply(x):
        x = layers.Conv2D(
            width,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x

    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        return x

    return apply


def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply


def build_model(
    img_size,
    img_channels,
    widths,
    has_attention,
    num_res_blocks=2,
    first_conv_channels=64,
    norm_groups=8,
    interpolation="nearest",
    activation_fn=keras.activations.swish,
):
    image_input = layers.Input(
        shape=(img_size, img_size, img_channels), name="image_input"
    )
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")

    x = layers.Conv2D(
        first_conv_channels,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(image_input)

    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)

    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(img_channels, (3, 3), padding="same", kernel_initializer=kernel_init(0.0))(x)

    return keras.Model([image_input, time_input], x, name="unet")


class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, timesteps, gdf_util, ema=0.999):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema

    def train_step(self, images):
        # 1. Get the batch size
        batch_size = tf.shape(images)[0]

        # 2. Sample timesteps uniformly
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
        )

        with tf.GradientTape() as tape:
            # 3. Sample random noise to be added to the images in the batch
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

            # 4. Diffuse the images with noise
            images_t = self.gdf_util.q_sample(images, t, noise)

            # 5. Pass the diffused images and time steps to the network
            pred_noise = self.network([images_t, t], training=True)

            # 6. Calculate the loss
            loss_obj = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            loss = loss_obj(noise, pred_noise) * (1.0 / tf.cast(batch_size, tf.float32))

        # 7. Get the gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 8. Update the weights of the network
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 9. Updates the weight values for the network with EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 10. Return loss values
        return {"loss": loss}

    def denoise_image(
        self,
        image, 
        denoise_timesteps,
        regularization_image=None,
        lambduh=0.005, 
        batch_size=16
    ):
        assert image.shape == (384, 384, 256)

        if denoise_timesteps > self.timesteps:
            raise ValueError(f'Given timestep parameter: {denoise_timesteps}. Cannot be greater than model: {self.timesteps}')

        image = np.moveaxis(image, -1, 0)
        image = np.expanddims(image, axis=-1)

        # t is [denoise_timesteps - 1, 0], e.g. 199 - 0
        for t in tqdm(
            reversed(range(0, denoise_timesteps)), 
            ncols=100, 
            desc=f'denoising {denoise_timesteps} times',
            bar_format='{l_bar}{bar}{r_bar}',
            total=denoise_timesteps
        ):
            tt = tf.cast(tf.fill(image.shape[0], t), dtype=tf.int64)
            pred_noise = self.ema_network.predict(
                [image, tt], verbose=0, batch_size=batch_size
            )
            image = self.gdf_util.p_sample(
                pred_noise, image, tt, clip_denoised=True
            )

            if regularization_image: 
                image = lambduh * regularization_image + (1 - lambduh) * image

        image = np.squeeze(image)
        image = np.moveaxis(image, 0, 1)

        return image

    def denoise_images_dct(
        self,
        images, 
        denoise_timesteps,
        regularization=True, 
        lambduh=0.005, 
        window_len=7, 
        keep_freq=[0],
        inference_batch_size=16
    ):
        if denoise_timesteps > self.timesteps:
            raise ValueError(f'Given timestep parameter: {denoise_timesteps}. Cannot be greater than model: {self.timesteps}')
        
        num_images = images.shape[0]
        img_shape  = (images.shape[1], images.shape[2])
        
        if regularization:
            zero_freq_img = np.zeros((num_images, ) + img_shape + (1, ), dtype='float32') # (num_images, 384, 384, 1)
            high_freq_img = np.zeros((num_images, ) + img_shape + (1, ) + (window_len * window_len, ), dtype='float32') # (num_images, 384, 384, 1, 49)

            for i in r_[:img_shape[0] - window_len]:
                for j in r_[:img_shape[1] - window_len]:
                    patch = images[:, i:i + window_len, j:j + window_len]
                    dct_patch = dct2(patch)
                    dct_vect = dct_patch.reshape((num_images, -1, 1)) # (256, 49, 1)
                    dct_vect = np.moveaxis(dct_vect, 1, -1)
                    
                    high_freq_img[:, i, j, :, :] = dct_vect
                    zero_freq_img[:, i, j, :]    = dct_vect[:, :, 0]
                    
            compound_freq_img = np.zeros(high_freq_img.shape[:-1], dtype='float32') # (num_images, 384, 384, 1)

            for kf in keep_freq: compound_freq_img += high_freq_img[:, :, :, :, kf]

            if denoise_timesteps == 1:
                plot2(images[128,:,:,:], compound_freq_img[128,:,:,:], title=['Original', f'Freq {keep_freq}'])
                plt.show()

        # t is [denoise_timesteps - 1, 0], e.g. 199 - 0
        for t in tqdm(
            reversed(range(0, denoise_timesteps)), 
            ncols=100, 
            desc=f'denoising {denoise_timesteps} times',
            bar_format='{l_bar}{bar}{r_bar}',
            total=denoise_timesteps
        ):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            pred_noise = self.ema_network.predict(
                [images, tt], verbose=0, batch_size=inference_batch_size
            )
            images = self.gdf_util.p_sample(
                pred_noise, images, tt, clip_denoised=True
            )

            #plot2(pred_noise[128,:,:], images[128,:,:], title=['pred noise', 'noise removed'])

            if regularization:
                before_images = np.copy(images)
                images = lambduh * compound_freq_img + (1 - lambduh) * images

        return images

    def generate_images(self, num_images=4, img_size=384, img_channels=1):
        # 1. Randomly sample noise (starting point for reverse process)
        samples = tf.random.normal(
            shape=(num_images, img_size, img_size, img_channels), dtype=tf.float32
        )
        # 2. Sample from the model iteratively
        for t in tqdm(
            reversed(range(0, self.timesteps)), 
            ncols=100, 
            desc='generating...', 
            total=self.timesteps
        ):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            pred_noise = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=num_images
            )
            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt, clip_denoised=True
            )
        # 3. Return generated samples
        return samples

    def plot_images(
        self, epoch=None, logs=None, num_rows=2, num_cols=4, figsize=(12, 5)
    ):
        """Utility to plot images using the diffusion model during training."""
        generated_samples = self.generate_images(num_images=num_rows * num_cols)
        generated_samples = (
            tf.clip_by_value(generated_samples * 127.5 + 127.5, 0.0, 255.0)
            .numpy()
            .astype(np.uint8)
        )

        fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
        for i, image in enumerate(generated_samples):
            if num_rows == 1:
                ax[i].imshow(image, cmap='gray')
                ax[i].axis("off")
            else:
                ax[i // num_cols, i % num_cols].imshow(image, cmap='gray')
                ax[i // num_cols, i % num_cols].axis("off")

        fig.suptitle(f'Epoch: {epoch}')
        plt.tight_layout()
        plt.savefig(f"./diffusion_images/ep{epoch}.png")
        plt.show()
        
    def save_model(
        self, epoch=None, logs=None
    ):
        if epoch % 10 == 0:
            path = './diffusion_models/'
            self.network.save_weights(os.path.join(path, f'diffusion_ep{epoch}.hd5'))
            self.ema_network.save_weights(os.path.join(path, f'ema_diffusion_ep{epoch}.hd5'))

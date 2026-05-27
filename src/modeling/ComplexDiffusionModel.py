import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import sys
import tensorflow as tf

from numpy import r_
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

from modeling.complex_utils import complex_conv2d, crelu
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


class ImageEmbedding(layers.Layer):
    def __init__(self, n_layers=4, feature_maps=[64,64,64,64], activation_fn=keras.activations.swish, **kwargs):
        super().__init__(**kwargs)
        assert n_layers == len(feature_maps)
        self.activation_fn = activation_fn

        self.conv_layers = []

        for filters in feature_maps:
            self.conv_layers.append(
                    layers.Conv2D(filters, kernel_size=3, padding='same')
            )
            self.conv_layers.append(
                    layers.MaxPooling2D(pool_size=(2, 2))
            )

        self.out_layers = [
                layers.Flatten(),
                layers.Dense(256, activation=activation_fn),
                layers.Dense(256)
        ]

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)

        for op in self.conv_layers:
            inputs = op(inputs)

        for op in self.out_layers:
            inputs = op(inputs)

        return inputs

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


def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(units, activation=activation_fn, kernel_initializer=kernel_init(1.0))(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply

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
    image_embedding=False
):
    image_input = layers.Input(shape=(img_size, img_size, img_channels), name="image_input")
    time_input  = keras.Input(shape=(), dtype=tf.int64, name="time_input")

    x = layers.Conv2D(
        first_conv_channels,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(image_input)

    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)

    if image_embedding:
        image_emb_input = layers.Input(shape=(img_size, img_size, img_channels), name='image_emb_input')
        iemb = ImageEmbedding(n_layers=4, feature_maps=[64, 64, 64, 64])(image_emb_input)
        temb = layers.Add()([iemb, temb])

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

    if image_embedding:
        return keras.Model([image_input, time_input, image_emb_input], x, name='unet_img_emb')
    else:
        return keras.Model([image_input, time_input], x, name="unet")


class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, timesteps, gdf_util, image_embedding=False, ema=0.999):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.image_embedding = image_embedding
        self.ema = ema
        self.last_loss = None

    def variable_noise(self, shape, partitions=6, dtype=np.float32):
        img_len = shape[-1]
        assert img_len % partitions == 0, 'Shape of image must be fully divisible by partitions.'

        res = np.random.normal(size=(img_len, img_len))
        stds = [ 1.0 + i * 0.25 for i in range(1, partitions) ]

        for std, p_ii in zip(stds, range(partitions - 1, 0, -1)):
            mid = img_len // 2
            part_len = (p_ii * img_len) // partitions
            offset = part_len // 2

            noise = np.random.normal(scale=std, size=(part_len, part_len))
            res[mid - offset: mid + offset, mid - offset: mid + offset] = noise

        return res

    def train_step(self, images, variable_noise=False):
        # 1. Get the batch size
        batch_size = tf.shape(images)[0]

        # 2. Sample timesteps uniformly
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
        )

        with tf.GradientTape() as tape:
            # 3. Sample random noise to be added to the images in the batch
            if variable_noise:
                res = np.stack([ variable_noise(tf.shape(images)[1:]) for i in range(batch_size) ])
                noise = tf.convert_to_tensor(res)
            else:
                noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

            # 4. Diffuse the images with noise
            images_t = self.gdf_util.q_sample(images, t, noise)

            # 5. Pass the diffused images and time steps to the network
            if self.image_embedding:
                inputs = [images_t, t, images_t]
            else:
                inputs = [images_t, t]
            pred_noise = self.network(inputs, training=True)

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

        self.last_loss = loss

        # 10. Return loss values
        return {"loss": loss}

    def denoise_image(
        self,
        image, 
        denoise_timesteps,
        regularization_image=None,
        batch_size=8,
    ):
        assert image.shape == (384, 384, 256)

        if denoise_timesteps > self.timesteps:
            raise ValueError(f'Given timestep parameter: {denoise_timesteps}. Cannot be greater than model: {self.timesteps}')

        # prepare image for denoising and regularization image too
        image = np.moveaxis(image, -1, 0)
        image = np.expand_dims(image, axis=-1)

        if regularization_image is not None and self.image_embedding:
            regularization_image = np.moveaxis(regularization_image, -1, 0)
            regularization_image = np.expand_dims(regularization_image, axis=-1)

        # t is [denoise_timesteps - 1, 0], e.g. 199 - 0
        for t in tqdm(
            reversed(range(0, denoise_timesteps)), 
            ncols=90, 
            desc=f'denoising {denoise_timesteps} times',
            bar_format='{l_bar}{bar}{r_bar}',
            total=denoise_timesteps
        ):
            tt = tf.cast(tf.fill(image.shape[0], t), dtype=tf.int64) # create vector of shape [256,] with all values set to t

            if regularization_image is not None and self.image_embedding:
                inputs = [image, tt, regularization_image]
            else:
                inputs = [image, tt]

            pred_noise = self.ema_network.predict(
                inputs, verbose=0, batch_size=batch_size
            )
            image = self.gdf_util.p_sample(
                pred_noise, image, tt, clip_denoised=True
            )

        image = np.squeeze(image)
        image = np.moveaxis(image, 0, -1)

        return image

    def generate_images(self, num_images=4, img_size=384, img_channels=1):
        # 1. Randomly sample noise (starting point for reverse process)
        samples = tf.random.normal(
            shape=(num_images, img_size, img_size, img_channels), dtype=tf.float32
        )
        # 2. Denoise the samples 1000 times
        for t in tqdm(
            reversed(range(0, self.timesteps)), 
            ncols=90, 
            desc='generating...', 
            total=self.timesteps
        ):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            if self.image_embedding:
                inputs = [samples, tt, samples]
            else:
                inputs = [samples, tt]
            pred_noise = self.ema_network.predict(
                inputs, verbose=0, batch_size=num_images
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
        if epoch % 5 == 0 or epoch < 15:
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

            if self.image_embedding:
                path = './diffusion_imgemb_images'
            else:
                path = './diffusion_images'

            fig.suptitle(f'Epoch: {epoch}')
            plt.tight_layout()
            plt.savefig(os.path.join(path, f"img_ep{epoch}.png"))
            #plt.show()

    def save_model(
        self, epoch=None, logs=None
    ):
        if self.image_embedding:
            path = './diffusion_imgemb_models'
        else:
            path = './diffusion_models'

        if epoch % 5 == 0:
            self.network.save_weights(os.path.join(path, f'diffusion_ep{epoch}.hd5'))
            self.ema_network.save_weights(os.path.join(path, f'ema_diffusion_ep{epoch}.hd5'))

        with open(os.path.join(path, 'loss_values.txt'), 'a') as f:
            last_loss = self.last_loss
            now = str(datetime.datetime.now())
            f.write(f'{now} ep{epoch} {last_loss}\n')

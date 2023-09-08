import tensorflow as tf
from tensorflow.keras.layers import Layer


def softshrink(x):
    alpha = tf.Variable(0.001, dtype='float32', trainable=True)
    sigma = tf.math.reduce_std(x)

    lower_th, upper_th = -1 * alpha * sigma, alpha * sigma

    z = _softshrink(x, lower_th, upper_th)

    return z


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

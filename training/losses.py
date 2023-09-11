import tensorflow as tf

class focal_loss(tf.keras.losses.Loss):

    def __init__(self, gamma_mu=2.0, gamma_nu=2.0, w_mu=1.0, w_nu=1.0):
        super().__init__()
        self.gamma_mu = gamma_mu
        self.gamma_nu = gamma_nu
        self.w_mu = w_mu
        self.w_nu = w_nu

    def __call__(self, y_true, y_pred, sample_weight=None):
        entropy = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        gamma = tf.where(y_true[:, 0] == 1, self.gamma_mu, self.gamma_nu)
        gamma = tf.expand_dims(gamma, axis=-1)
        focal_weight = tf.reduce_sum(y_true * tf.math.pow((1 - y_pred), gamma), axis=-1)
        entropy = tf.math.multiply(entropy, focal_weight)
        if sample_weight is not None:
            sample_weight = tf.squeeze(sample_weight)
            sample_weight = tf.where(y_true[:, 0] == 1, self.w_mu * sample_weight, self.w_nu * sample_weight)
        else:
            sample_weight = tf.where(y_true[:, 0] == 1, self.w_mu, self.w_nu)
        entropy = tf.math.reduce_mean(tf.math.multiply(entropy, sample_weight))
        return entropy
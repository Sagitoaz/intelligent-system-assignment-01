"""Keras definitions matching the NumPy architectures (NLC layout)."""

import tensorflow as tf


class TFCNN(tf.keras.Model):
    def __init__(self, input_channels, output_dim, *, improved=False):
        super().__init__()
        self.improved = improved
        self.conv1 = tf.keras.layers.Conv1D(8, 3, padding="same", name="conv1")
        self.conv2 = tf.keras.layers.Conv1D(16, 3, padding="same", name="conv2")
        self.pool = tf.keras.layers.MaxPooling1D(2, 2, name="pool")
        if improved:
            self.conv3 = tf.keras.layers.Conv1D(32, 3, padding="same", name="conv3")
            self.hidden = tf.keras.layers.Dense(16, name="hidden")
        self.output_layer = tf.keras.layers.Dense(output_dim, name="output")

    def call(self, x, mask=None):
        x = tf.nn.relu(self.conv1(x))
        x = tf.nn.relu(self.conv2(x))
        x = self.pool(x)
        pooled_mask = None
        if mask is not None:
            usable = (tf.shape(mask)[1] // 2) * 2
            pooled_mask = tf.reduce_any(tf.reshape(mask[:, :usable], (tf.shape(mask)[0], -1, 2)), axis=2)
        if self.improved:
            x = tf.nn.relu(self.conv3(x))
        if pooled_mask is None:
            x = tf.reduce_max(x, axis=1)
        else:
            minimum = tf.cast(-1e30, x.dtype)
            x = tf.reduce_max(tf.where(pooled_mask[:, :, None], x, minimum), axis=1)
            x = tf.where(tf.reduce_any(pooled_mask, axis=1, keepdims=True), x, tf.zeros_like(x))
        if self.improved:
            x = tf.nn.relu(self.hidden(x))
        return self.output_layer(x)

    @property
    def trainable_layer_count(self):
        return 5 if self.improved else 3


def build_tf_cnn(length, input_channels, output_dim, *, improved=False):
    model = TFCNN(input_channels, output_dim, improved=improved)
    model(tf.zeros((1, length, input_channels)))
    return model


class TFTextCNN(tf.keras.Model):
    def __init__(self, vocabulary_size, sequence_length, output_dim=2, *, improved=False):
        super().__init__()
        # The CNN propagates its explicit token mask after pooling; Keras'
        # implicit mask is disabled because Conv1D does not propagate it.
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, 16, mask_zero=False, name="embedding")
        self.cnn = TFCNN(16, output_dim, improved=improved)
        self(tf.zeros((1, sequence_length), dtype=tf.int32))

    def embed_tokens(self, ids):
        ids = tf.convert_to_tensor(ids)
        embedded = self.embedding(ids)
        # Zero PAD values in the computation graph; its embedding row receives
        # no gradient and cannot create artificial convolution activations.
        return tf.where(tf.not_equal(ids, 0)[:, :, None], embedded, tf.zeros_like(embedded))

    def call(self, ids):
        return self.cnn(self.embed_tokens(ids), tf.not_equal(ids, 0))


def build_tf_text_cnn(vocabulary_size, sequence_length, output_dim=2, *, improved=False):
    return TFTextCNN(vocabulary_size, sequence_length, output_dim, improved=improved)


class TFHouseFieldEncoder(tf.keras.layers.Layer):
    def __init__(self, state_size, status_size):
        super().__init__()
        self.state_embedding = tf.keras.layers.Embedding(state_size, 8, mask_zero=False)
        self.status_embedding = tf.keras.layers.Embedding(status_size, 8, mask_zero=False)

    def build(self, input_shape):
        self.numeric_weight = self.add_weight(shape=(4, 8), initializer=tf.keras.initializers.RandomNormal(stddev=0.1), name="numeric_weight")
        self.numeric_bias = self.add_weight(shape=(4, 8), initializer="zeros", name="numeric_bias")

    def call(self, inputs):
        numeric, state_ids, status_ids = inputs
        numeric_tokens = numeric[:, :, None] * self.numeric_weight[None, :, :] + self.numeric_bias[None, :, :]
        state = self.state_embedding(tf.squeeze(state_ids, axis=1))[:, None, :]
        status = self.status_embedding(tf.squeeze(status_ids, axis=1))[:, None, :]
        return tf.concat([state, status, numeric_tokens], axis=1)


class TFHouseCNN(tf.keras.Model):
    def __init__(self, state_size, status_size, *, improved=False):
        super().__init__()
        self.encoder = TFHouseFieldEncoder(state_size, status_size)
        self.cnn = TFCNN(8, 1, improved=improved)

    def call(self, inputs):
        return self.cnn(self.encoder(inputs))


def build_tf_house_cnn(state_size, status_size, *, improved=False):
    model = TFHouseCNN(state_size, status_size, improved=improved)
    model([tf.zeros((1, 4)), tf.zeros((1, 1), dtype=tf.int32), tf.zeros((1, 1), dtype=tf.int32)])
    return model

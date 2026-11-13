"""A minimal custom Keras layer: a trainable linear projection (no bias).

MyLayer(output_dim) computes  y = x @ kernel,  where kernel has shape
(input_features, output_dim) and is created lazily in build().
"""

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="kerasExamples")
class MyLayer(keras.layers.Layer):

    def __init__(self, output_dim=12, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        # Create the trainable weight once the input size is known.
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.output_dim),
            initializer="uniform",
            trainable=True,
        )

    def call(self, x):
        return ops.matmul(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.output_dim)

    def get_config(self):
        # Needed so the layer can be rebuilt when a saved model is loaded.
        config = super().get_config()
        config.update({"output_dim": self.output_dim})
        return config

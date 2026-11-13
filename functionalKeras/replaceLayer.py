"""Replace an intermediate layer in an existing Keras model.

Builds a small CNN, then rebuilds its graph with the layer at a given
index swapped for a new one: first a bias-free Conv2D, then an
identity Lambda. The other layers are reused, so they keep their weights.

Idea from: https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model

Run:  python replaceLayer.py
"""

import keras
from keras import layers


def keras_simple_model():
    inputs = keras.Input((28, 28, 1))
    x = layers.Conv2D(4, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(4, (3, 3), padding="same", name="conv2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(x)

    x = layers.Conv2D(8, (3, 3), padding="same", name="conv3")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(8, (3, 3), padding="same", name="conv4")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10)(x)
    x = layers.Activation("softmax")(x)
    return keras.Model(inputs=inputs, outputs=x)


def replace_intermediate_layer_in_keras(model, layer_id, new_layer):
    """Return a new model with model.layers[layer_id] replaced by new_layer.

    Works for linear (chain-shaped) models, where each layer feeds the next.
    """
    model_layers = model.layers
    inputs = model_layers[0].output  # layers[0] is the InputLayer
    x = inputs
    for i in range(1, len(model_layers)):
        x = new_layer(x) if i == layer_id else model_layers[i](x)
    return keras.Model(inputs=inputs, outputs=x)


def report(model, plot_file):
    model.summary()
    print(f"[INFO] Total number of parameters in model: {model.count_params()}")
    try:
        keras.utils.plot_model(model, to_file=plot_file, show_shapes=True, show_layer_names=True)
    except (ImportError, OSError) as err:
        print(f"[WARN] Skipped plot_model: {err}")


if __name__ == "__main__":
    model = keras_simple_model()
    report(model, "modelBeforeLayerReplacement.png")

    # Layer 3 is conv2. Replace it with a Conv2D that has no bias (4 fewer parameters).
    new_layer = layers.Conv2D(4, (3, 3), padding="same", name="conv2_repl", use_bias=False)
    model = replace_intermediate_layer_in_keras(model, 3, new_layer)
    report(model, "modelAfterLayerReplacement1.png")

    # Replace it again, this time with an identity Lambda (no parameters).
    new_layer = layers.Lambda(lambda t: t * 1, name="identity")
    model = replace_intermediate_layer_in_keras(model, 3, new_layer)
    report(model, "modelAfterLayerReplacement2.png")

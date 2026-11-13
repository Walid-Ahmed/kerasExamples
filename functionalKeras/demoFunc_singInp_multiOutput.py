"""Functional API: one input, two outputs.

A shared Dense layer feeds two separate heads, each with its own
softmax output and its own loss.

Run:  python demoFunc_singInp_multiOutput.py
"""

import numpy as np
import keras
from keras import layers

inputs = keras.Input(shape=(12,), name="input")
shared = layers.Dense(64, activation="relu", name="shared")(inputs)

head1 = layers.Dense(64, activation="relu", name="head_1")(shared)
head2 = layers.Dense(64, activation="relu", name="head_2")(shared)

out1 = layers.Dense(10, activation="softmax", name="out_1")(head1)
out2 = layers.Dense(10, activation="softmax", name="out_2")(head2)

model = keras.Model(inputs=inputs, outputs=[out1, out2], name="single_in_multi_out")

# With several outputs, losses and metrics are given per output name.
model.compile(optimizer="rmsprop",
              loss={"out_1": "categorical_crossentropy", "out_2": "categorical_crossentropy"},
              metrics={"out_1": ["accuracy"], "out_2": ["accuracy"]})
model.summary()

n = 256
x = np.random.rand(n, 12).astype("float32")
y1 = keras.utils.to_categorical(np.random.randint(10, size=n), 10)
y2 = keras.utils.to_categorical(np.random.randint(10, size=n), 10)
model.fit(x, {"out_1": y1, "out_2": y2}, epochs=1, batch_size=32)

# Drawing the graph needs pydot and Graphviz; skip it if they are missing.
try:
    keras.utils.plot_model(model, to_file="singInp_multiOut.png", show_shapes=True, show_layer_names=True)
except (ImportError, OSError) as err:
    print(f"[WARN] Skipped plot_model: {err}")

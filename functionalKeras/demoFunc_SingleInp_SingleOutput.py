"""Functional API: one input, one output.

A plain stack of Dense layers built with the functional API, the
functional equivalent of a Sequential model.

Run:  python demoFunc_SingleInp_SingleOutput.py
"""

import numpy as np
import keras
from keras import layers

# Input -> Dense(64) -> Dense(64) -> Dense(10, softmax)
inputs = keras.Input(shape=(784,), name="input")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
predictions = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=predictions, name="single_in_single_out")
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# Train for one epoch on random data to show the expected shapes.
x_train = np.random.rand(256, 784).astype("float32")
y_train = keras.utils.to_categorical(np.random.randint(10, size=256), 10)
model.fit(x_train, y_train, epochs=1, batch_size=32)

# Drawing the graph needs pydot and Graphviz; skip it if they are missing.
try:
    keras.utils.plot_model(model, to_file="single_Inp_Single_Output.png", show_shapes=True, show_layer_names=True)
except (ImportError, OSError) as err:
    print(f"[WARN] Skipped plot_model: {err}")

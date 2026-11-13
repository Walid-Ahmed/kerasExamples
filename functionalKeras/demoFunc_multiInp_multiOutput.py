"""Functional API: two inputs, two outputs.

A main input and an auxiliary input each have their own branch. The
auxiliary branch produces an auxiliary output early, and both branches
are merged to produce the main output.

Run:  python demoFunc_multiInp_multiOutput.py
"""

import numpy as np
import keras
from keras import layers

main_input = keras.Input(shape=(12,), name="main_input")
aux_input = keras.Input(shape=(12,), name="aux_input")

# Auxiliary branch, with its own output.
aux = layers.Dense(12, activation="relu", name="aux_dense")(aux_input)
aux_output = layers.Dense(1, activation="sigmoid", name="aux_output")(aux)

# Main branch.
main = layers.Dense(16, activation="relu", name="main_dense_1")(main_input)
main = layers.Dense(8, activation="relu", name="main_dense_2")(main)

# Merge both branches for the main output.
merged = layers.Concatenate(name="concat")([aux, main])
main_output = layers.Dense(10, activation="softmax", name="main_output")(merged)

model = keras.Model(inputs=[main_input, aux_input],
                    outputs=[main_output, aux_output],
                    name="multi_in_multi_out")

# loss_weights lets the auxiliary loss count for less than the main one.
model.compile(optimizer="rmsprop",
              loss={"main_output": "categorical_crossentropy",
                    "aux_output": "binary_crossentropy"},
              loss_weights={"main_output": 1.0, "aux_output": 0.2},
              metrics={"main_output": ["accuracy"], "aux_output": ["accuracy"]})
model.summary()

# Inputs and targets can be passed as dicts keyed by layer name.
n = 256
x_main = np.random.rand(n, 12).astype("float32")
x_aux = np.random.rand(n, 12).astype("float32")
y_main = keras.utils.to_categorical(np.random.randint(10, size=n), 10)
y_aux = np.random.randint(2, size=(n, 1)).astype("float32")

model.fit({"main_input": x_main, "aux_input": x_aux},
          {"main_output": y_main, "aux_output": y_aux},
          epochs=1, batch_size=32)

# Drawing the graph needs pydot and Graphviz; skip it if they are missing.
try:
    keras.utils.plot_model(model, to_file="multiInp_multiOut.png", show_shapes=True, show_layer_names=True)
except (ImportError, OSError) as err:
    print(f"[WARN] Skipped plot_model: {err}")

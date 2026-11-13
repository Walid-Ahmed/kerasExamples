"""Functional API: two inputs, one output.

Each input goes through its own Dense branch; the branches are merged
with Concatenate and feed a single softmax output.

Run:  python demoFunc_MulInp_SingleOutput.py
"""

import numpy as np
import keras
from keras import layers

input1 = keras.Input(shape=(12,), name="input_1")
input2 = keras.Input(shape=(64,), name="input_2")

branch1 = layers.Dense(64, activation="relu", name="branch_1")(input1)
branch2 = layers.Dense(64, activation="relu", name="branch_2")(input2)

merged = layers.Concatenate(name="concat")([branch1, branch2])
predictions = layers.Dense(10, activation="softmax", name="predictions")(merged)

model = keras.Model(inputs=[input1, input2], outputs=predictions, name="multi_in_single_out")
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# A multi-input model takes a list of arrays, in the same order as `inputs`.
n = 256
x1 = np.random.rand(n, 12).astype("float32")
x2 = np.random.rand(n, 64).astype("float32")
y = keras.utils.to_categorical(np.random.randint(10, size=n), 10)
model.fit([x1, x2], y, epochs=1, batch_size=32)

# Drawing the graph needs pydot and Graphviz; skip it if they are missing.
try:
    keras.utils.plot_model(model, to_file="multi_Inp_Single_Output.png", show_shapes=True, show_layer_names=True)
except (ImportError, OSError) as err:
    print(f"[WARN] Skipped plot_model: {err}")

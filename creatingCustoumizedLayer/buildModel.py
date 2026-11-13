"""Build a model that uses the custom MyLayer and save it to model.keras.

Run:  python buildModel.py
"""

import keras
from keras import layers

from myLayer import MyLayer

model = keras.Sequential([
    keras.Input(shape=(784,)),
    layers.Dense(32, activation="relu"),
    MyLayer(output_dim=5),
    layers.Dense(1, activation="sigmoid"),
])
model.summary()

file_name = "model.keras"
model.save(file_name)
print(f"[INFO] Model saved to {file_name}")

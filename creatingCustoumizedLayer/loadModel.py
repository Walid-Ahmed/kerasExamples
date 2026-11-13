"""Load the model saved by buildModel.py, including its custom MyLayer.

Run buildModel.py first, then:  python loadModel.py
"""

import numpy as np
import keras

# Importing MyLayer registers it, so Keras knows how to rebuild it.
# Passing custom_objects as well makes the dependency explicit.
from myLayer import MyLayer

model = keras.models.load_model("model.keras", custom_objects={"MyLayer": MyLayer})
print("[INFO] Model with custom layer loaded successfully")
model.summary()

prediction = model.predict(np.random.rand(1, 784).astype("float32"), verbose=0)
print(f"[INFO] Prediction on a random sample: {prediction[0][0]:.4f}")

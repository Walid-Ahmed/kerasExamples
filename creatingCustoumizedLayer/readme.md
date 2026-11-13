# Building your own Keras layer

`MyLayer` in [`myLayer.py`](myLayer.py) is a custom trainable layer that computes `y = x @ kernel`.

## Usage

```bash
python buildModel.py   # build a model that uses MyLayer and save it to model.keras
```

```bash
python loadModel.py    # load model.keras, custom layer included, and run a prediction
```

See the [main README](../README.md#creating-a-custom-layer) for how the layer is written.

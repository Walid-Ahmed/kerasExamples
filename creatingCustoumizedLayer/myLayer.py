from keras import backend as K
from keras.layers import Layer

from keras.models import Sequential
from keras.layers import Dense, Activation

class MyLayer(Layer):

    def __init__(self, output_dim=12, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


    def get_config(self):
      config = {
      'output_dim': self.output_dim
      }
      base_config = super(MyLayer, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))



# python demoFunc_singInp_multiOutput.py

from keras.layers import Input, Dense
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.layers import Concatenate



##################Multiple  Input   , single  output ###########################
# This returns a tensor
input1 = Input(shape=(12,))

# a layer instance is callable on a tensor, and returns a tensor
output_1 = Dense(64, activation='relu')(input1)

output_2 = Dense(64, activation='relu')(output_1)
output_3 = Dense(64, activation='relu')(output_1)



predictions1 = Dense(10, activation='softmax')(output_2)
predictions2 = Dense(10, activation='softmax')(output_3)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=input1, outputs=[predictions1,predictions2])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


plotFileName="singInp_multiOut.png"
from keras.utils import plot_model
plot_model(model, to_file=plotFileName ,show_shapes=True)


##################################################################


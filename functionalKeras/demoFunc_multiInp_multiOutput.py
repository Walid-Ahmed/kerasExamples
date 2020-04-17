
# python demoFunc_multiInp_multiOutput.py

from keras.layers import Input, Dense
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.layers import Concatenate



##################Multiple  Input   , Multiple  output ###########################
# This returns a tensor


main_input = Input(shape=(12,))
auxiliary_input = Input(shape=(12,))


output_1 = Dense(12, activation='relu')(auxiliary_input)
output_1_aux = Dense(12, activation='relu')(output_1)  # auxillary output

output_2 = Dense(16, activation='relu')(main_input)
output_3 = Dense(8, activation='relu')(output_2)

output4=Concatenate()([output_1,output_3])



output4 = Dense(10, activation='softmax')(output4)
output4 = Dense(10, activation='softmax')(output4)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=[main_input,auxiliary_input] , outputs=[output_1_aux,output4])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


plotFileName="multiInp_multiOut.png"
from keras.utils import plot_model
plot_model(model, to_file=plotFileName ,show_shapes=True)


##################################################################

import numpy as np

input1 = np.round(np.abs(np.random.rand(12, 100) * 100))
input2 = np.random.randn(12, 5)
output1 = np.random.randn(12, 1)
output2 = np.random.randn(12, 1)





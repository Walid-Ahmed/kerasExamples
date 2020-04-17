
# python demoFunc_MulInp_SingleOutput.py

from keras.layers import Input, Dense
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.layers import Concatenate



##################Multiple  Input   , single  output ###########################
# This returns a tensor
input1 = Input(shape=(12,))
input2 = Input(shape=(64,))

# a layer instance is callable on a tensor, and returns a tensor
output_1 = Dense(64, activation='relu')(input1)
output_2 = Dense(64, activation='relu')(input2)

conc=Concatenate()([output_1,output_2])

predictions = Dense(10, activation='softmax')(conc)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=[input1,input2], outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


plotFileName="multi_Inp_Single_Output.png"
from keras.utils import plot_model
plot_model(model, to_file=plotFileName)


##################################################################


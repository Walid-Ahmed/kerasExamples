#python demoFunc_SingleInp_SingleOutput.py

from keras.layers import Input, Dense
from keras.models import Model
from keras.preprocessing.image import load_img



# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
output_1 = Dense(64, activation='relu')(inputs)
output_2 = Dense(64, activation='relu')(output_1)
predictions = Dense(10, activation='softmax')(output_2)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



plotFileName="single_Inp_Single_Output.png"
from keras.utils import plot_model
plot_model(model, to_file=plotFileName ,show_shapes=True)

#model.fit(data, labels)  # starts training



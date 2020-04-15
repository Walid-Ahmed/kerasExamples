from keras import backend as K
from keras.layers import Layer

from keras.models import Sequential
from keras.layers import Dense, Activation
from myLayer  import  MyLayer




myLayer=MyLayer(12)
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))     
model.add(myLayer)   
model.add(Dense(1, activation='sigmoid'))

model.summary()

fileNameSaveModel="model.h5"
model.save(fileNameSaveModel)
print("[INFO]  Model saved  to file {} ".format(fileNameSaveModel))

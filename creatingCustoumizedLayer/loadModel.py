# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
from myLayer  import  MyLayer


print("[INFO] myLayer loaded sucessfully")
# load model
model = load_model('model.h5',  custom_objects={'MyLayer':MyLayer})
# summarize model.
model.summary()
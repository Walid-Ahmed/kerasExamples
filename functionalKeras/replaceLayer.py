 #python replaceLayer.py
 #SRC https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model


'''
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense,  GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D, Activation
'''


from keras.layers import Conv2D
from keras.models import Model
from keras.layers import Input, Dense,  GlobalAveragePooling2D
from keras.layers import MaxPooling2D, Activation


def keras_simple_model():


    inputs1 = Input((28, 28, 1))
    x = Conv2D(4, (3, 3), activation=None, padding='same', name='conv1')(inputs1)
    x = Activation('relu')(x)
    x = Conv2D(4, (3, 3), activation=None, padding='same', name='conv2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    x = Conv2D(8, (3, 3), activation=None, padding='same', name='conv3')(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), activation=None, padding='same', name='conv4')(x)


    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation=None)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=inputs1, outputs=x)
    return model



def replace_intermediate_layer_in_keras(model, layer_id, new_layer):

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        else:
            x = layers[i](x)

    new_model = Model(input=layers[0].input, output=x)
    return new_model

if __name__ == '__main__':
	model = keras_simple_model()
	model.summary()

	plotFileName="modelBeforeLayerReplacement.png"
	from keras.utils import plot_model
	plot_model(model, to_file=plotFileName ,show_shapes=True)

	new_layer= Conv2D(4, (3, 3), activation=None, padding='same', name='conv2_repl', use_bias=False)
	model = replace_intermediate_layer_in_keras(model, 3,new_layer)
	model.summary()


	plotFileName="modelAfterLayerReplacement.png"
	from keras.utils import plot_model
	plot_model(model, to_file=plotFileName ,show_shapes=True)
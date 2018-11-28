from keras.layers import Input, Dense, LeakyReLU
from keras import initializers
from keras.models import Model
import keras.backend as K
def GenerativeNet(input_data):
    Input_Data = Input(input_data)
    X = Dense(256, input_dim=K.shape(Input_Data), kernel_initializer=initializers.RandomNormal(stddev=0.02))(Input_Data)
    X = LeakyReLU(alpha=0.2)(X)
    X = Dense(512, input_dim=K.shape(X), kernel_initializer=initializers.RandomNormal(stddev=0.02))(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = Dense(1024, input_dim=K.shape(X), kernel_initializer=initializers.RandomNormal(stddev=0.02))(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = Dense(784, input_dim=K.shape(X), activation='tanh')(X)
    model = Model(inputs=Input_Data, outputs=X, name='GenerativeNet')
    return model
from keras.layers import Input, Dense, LeakyReLU, Dropout
from keras import initializers
from keras.models import Model
import keras.backend as K
def DiscriminativeNet(input_data):
    Input_Data = Input(input_data)
    X = Dense(1024, input_dim=K.shape(Input_Data), kernel_initializer=initializers.RandomNormal(stddev=0.02))(Input_Data)
    X = LeakyReLU(alpha=0.2)(X)
    X = Dropout(rate = 0.3)(X)
    X = Dense(512, input_dim=K.shape(X))(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = Dropout(rate = 0.3)(X)
    X = Dense(256, input_dim=K.shape(X))(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = Dropout(rate = 0.3)(X)
    X = Dense(1, input_dim=K.shape(X), activation='sigmoid')(X)
    model = Model(inputs=Input_Data, outputs=X, name='DiscriminativeNet')
    return model
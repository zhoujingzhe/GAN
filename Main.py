import numpy as np
from keras.datasets import mnist
from keras.optimizers import Adam
from GAN import GenerativeAdversarialNet
from DiscriminativeNet import DiscriminativeNet
from GenerativeNet import GenerativeNet
from Util import randomDim, train
print('randomDim', randomDim)

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train.reshape(60000, 784)

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)
G_net = GenerativeNet(input_data=(randomDim,))
D_net = DiscriminativeNet(input_data=(784,))
G_net.compile(optimizer=adam, loss='binary_crossentropy')
D_net.compile(optimizer=adam, loss='binary_crossentropy')
Gan = GenerativeAdversarialNet(discriminator=D_net, generator=G_net)
Gan.compile(optimizer=adam, loss='binary_crossentropy')
train(epochs=200, batchSize=128, discriminator=D_net, generator=G_net,gan=Gan, X_train=X_train)







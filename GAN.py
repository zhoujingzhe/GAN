from keras.models import Model
from keras.layers import Input
from Util import randomDim

# Combined network
def GenerativeAdversarialNet(discriminator, generator):
    discriminator.trainable = False
    ganInput = Input(shape=(randomDim,))
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    return gan


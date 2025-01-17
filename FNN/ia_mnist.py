import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import h5py

# Get Data + Prepare Data
(x_train,y_train), (x_test,y_test) = mnist.load_data()

# Préparation de la transformation des tensors en matrices
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

# Allocation du stockage des décimaux
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Transformation de 0 à 255 en 0 à 1
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Model

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(784,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
# Couche visible de sortie (valeur entre 0 et 9)
model.add(Dense(10, activation='softmax'))

# Affiche la sortie dans la console
model.summary()

# On compile le modèle
model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
# Entrainement du modèle
model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1, validation_data=(x_test, y_test))

# On sauvegarde le modèle
model.save("model_mnist.h5")

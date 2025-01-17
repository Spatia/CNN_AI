# On importe les bibliothèques
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import h5py
from glob import glob
from tqdm import tqdm
import cv2


# On définit les variables
x_train = []
y_train = []
x_test = []
y_test = []

# On importe la dataset
dossier_train = glob("dataset/training/*")
dossier_test = glob("dataset/validation/*")
nbImagesTrain = len(glob("dataset/training/*/*.jpg"))
nbImagesTest = len(glob("dataset/validation/*/*.jpg"))

# Boucle qui va parourir tout le dossier de la dataset
for dossier in dossier_train :
	label = int(dossier[-1])
	print(label)
	noms_images = glob(dossier+"/*.jpg")
	for nom_courant in tqdm(noms_images) :
		y_train.append(label)
		image = cv2.imread(nom_courant, cv2.IMREAD_COLOR)
		image = cv2.resize(image,(512,512))
		image = image.astype("float32")
		image /= 255
		x_train.append(image)

for dossier in dossier_test :
	label = int(dossier[-1])
	print(label)
	noms_images = glob(dossier+"/*.jpg")
	for nom_courant in tqdm(noms_images) :
		y_test.append(label)
		image = cv2.imread(nom_courant, cv2.IMREAD_COLOR)
		image = cv2.resize(image,(512,512))
		image = image.astype("float32")
		image /= 255
		x_test.append(image)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(x_train.shape)
print(x_test.shape)
x_train = x_train.reshape(nbImagesTrain, 512, 512, 3)
x_test = x_test.reshape(nbImagesTest, 512, 512, 3)

# Modèle
model = Sequential()

model.add(Conv2D(64, kernel_size=(3,3), activation="relu", input_shape=(512, 512, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=(3,3), activation="relu"))
model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
model.add(Flatten())

model.add(Dense(32, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1, validation_data=(x_test, y_test))

model.save("model_cnn_monkey.h5")
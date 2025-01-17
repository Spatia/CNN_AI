from tkinter import *
from tkinter import filedialog
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import h5py
from glob import glob
from tqdm import tqdm
import cv2

root = Tk()
root.withdraw()
root.filename = filedialog.askopenfilename(filetypes=(("Image JPEG", "*.jpg"),("Tous les fichiers", "*.*")))

# Charge le modèle de réseau de neurone
model = load_model("/Users/ck192/Desktop/IA/CNN/model_cnn_monkey.h5")

# Ouvrir l'image
nom_img = root.filename
image = cv2.imread(nom_img, cv2.IMREAD_COLOR)

image = cv2.resize(image,(512,512))

# Reshape en une liste
image = image.reshape(1, 512, 512, 3)

# Passage  en float32
image = image.astype("float32")
# Divise par 255
image /= 255

# Prédictions
prediction = model.predict(image)
# Affiche la possibilité la plus probable (Quel est le chiffre que ça a le plus de chance d'être)
prediction = np.argmax(prediction)

print("#################################################")
print("Pour l'image : " + nom_img + " la prediction est : ")
print(prediction)
print("#################################################")

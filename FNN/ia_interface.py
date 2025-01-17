import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense
import cv2
import numpy as np

# Charge le modèle de réseau de neurone
model = load_model("C:/Users/antoi/Progs/Intelligence Artificielle MagicMakers/model_mnist.h5")

# Ouvrir l'image
nom_img = "chiffre.png"
image = cv2.imread(nom_img, cv2.IMREAD_GRAYSCALE)

# Reshape en une liste
image = image.reshape((1,28*28))

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

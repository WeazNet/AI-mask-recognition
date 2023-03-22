from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import cv2
import os
import resource
import sys
import numpy as np

# Initial Learning Rate 
INIT_LR = 1e-3
# Nombre de données d'entraînement pour un cycle.
EPOCHS = 25
# Batch size : la taille du paquêt traité
BS = 8

DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'
TRAIN_PATH = DIR + '/train/'
VALIDATION_PATH = DIR + '/validation/'

class TrainMask:

    def __init__(self, data = [], labels = []):
        self.data = data
        self.labels = labels

        self.hydrate()

        self.labels = np.array(labels)
        self.data = np.array(data) / 255.0

        lb = LabelBinarizer()
        self.labels = lb.fit_transform(labels)
        self.labels = to_categorical(labels)

    def hydrate(self):
        for file in os.listdir(TRAIN_PATH):
            file_name, file_extension = os.path.splitext(file)
            self.labels.append(int(file_name.split("_")[1]))

            image = cv2.imread(TRAIN_PATH+file_name+file_extension)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))

            self.data.append(image)

    # INFO - Entraînement de la reconnaissance du port du masque et création d'un fichier contenant les poids idéaux
    def train(self):
        (trainX, testX, trainY, testY) = train_test_split(self.data, self.labels,test_size=0.2, stratify=self.labels, random_state=5)
        model = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 1)))

        x = model.output
        # Conversion des sorties en un tableau 1D
        x = Flatten(name="flatten")(x)
        # Connexion de 512 neuronnes en activation RELU: qui force les valeurs max(x,0)
        x = Dense(512, activation="relu")(x)
        # Définit aléatoirement les éléments sur zéro pour éviter le surajustement
        x = Dropout(0.5)(x)
        # Softmax convertit un vecteur de valeurs en une distribution de probabilité.
        # Les éléments du vecteur de sortie sont dans la plage (0, 1) et de somme à 1.
        x = Dense(2, activation="softmax")(x)

        new_model = Model(inputs=model.input, outputs=x)

        # Pour changer la valeur par défaut du Keras
        for layer in model.layers:
            layer.trainable = False

        #Compilation du modèle
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        new_model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

        datagen = ImageDataGenerator(
        rotation_range=15,
        fill_mode="nearest")

        print("Apprentissage...")

        new_model.fit(datagen.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) / BS, epochs=EPOCHS)

        new_model.save("mask_learn.model", save_format="h5")
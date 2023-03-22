import os
import cv2
import numpy as np
import tensorflow as tf

class MaskRecognition:

    def __init__(self):
        self.model = tf.keras.models.load_model("mask_learn_tiny.model")


    # INFO - Recupère les visages (de la méthode auto-crop-image) et détermine selon les poids du modèle
    # si oui ou non la personne porte un masque (avec la methode predict), elle affiche alors un cadre autour de la tête
    # avec le résultat de la prédiction 
    # PARAMS - L'argument image est de taille 160x160 
    def scan(self, images, faces, frame):
        n = len(images)
        withoutMask = []
        withMask = []
        for i in range(n):
            image = cv2.resize(images[i], (224, 224))
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            predict = self.model.predict(image)

            withoutMask.append(predict[0][0])
            withMask.append(predict[0][1])

            (x, y, w, h) = faces[i]
            if withoutMask[i] > withMask[i]:
                cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 0, 255), 2)
                cv2.putText(frame, "PAS DE MASQUE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x,y), (x+w, y+h),(0, 255, 0), 2)
                cv2.putText(frame, "OK", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
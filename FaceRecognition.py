import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2
from scipy.io import loadmat
import dlib
from imutils import face_utils

DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'
TRAIN_PATH = DIR + '/train/'
DOTS_SIZE = 2

class FaceRecognition:
    def __init__(self):
        data = loadmat('vgg-face.mat', matlab_compatible=False, struct_as_record=False)
        self.l = data['layers']

    # INFO - Récuperation des visages pour la détection du port du masque
    # PARAMS - L'argument image est de taille 160x160 
    def auto_crop_image(self, image):
        if image is not None:
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            number = len(faces)
            print(f"{number} visage(s) reconnu(s) !")
            resized = []
            for (x, y, w, h) in faces:
                center_x = x+w/2
                center_y = y+h/2
                width, height, _ = image.shape
                dim = min(max(w,h),width, height)
                box = [center_x-dim/2, center_y-dim/2, center_x+dim/2, center_y+dim/2]
                box = [int(x) for x in box]
                if box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height:
                    temp = image[box[1]:box[3],box[0]:box[2]]
                    resized.append(cv2.resize(temp, (160,160), interpolation = cv2.INTER_AREA))
            return faces, resized

    # PARAMS - L'argument image est de taille 160x160
    # Le nom de l'image comporte déjà le suffixe attendu
    def save(self, image, image_name):
        cv2.imwrite(TRAIN_PATH+image_name+'.jpg', image)
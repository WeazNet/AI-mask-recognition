import os
import cv2
from FaceRecognition import FaceRecognition
from MaskRecognition import MaskRecognition

class Camera:

    def __init__(self, name, faceDetection, maskDetection, exitKeyCode = 27):
        self.name = name
        self.exitKeyCode = exitKeyCode #ESC
        self.faceDetection = faceDetection
        self.maskDetection = maskDetection

    # INFO - Lancer la caméra et activer la détection du port du masque
    def launch(self):
        cv2.namedWindow(self.name)
        vc = cv2.VideoCapture(0)
        faceRec = FaceRecognition()
        maskRec = MaskRecognition()

        while vc.isOpened():
            _, frame = vc.read()

            if self.faceDetection:
                faces, resized = faceRec.auto_crop_image(frame)
                # La détection du visage est nécessaire au fonctionnement de la détection du masque
                if self.maskDetection:
                    maskRec.scan(resized, faces, frame)

            key = cv2.waitKey(100)
            cv2.imshow(self.name, frame)

            if key == self.exitKeyCode:
                break

        del faceRec
        del maskRec
        cv2.destroyWindow(self.name)
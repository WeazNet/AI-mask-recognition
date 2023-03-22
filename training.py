import os
import cv2

#from TrainMask import TrainMask
from FaceRecognition import FaceRecognition

DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'
NOCONVERT_PATH = DIR + '/noConvert/'

# INFO - Fonction auxiliaire servant à, à partir d'une image couper les visages trouvés, les enregistrer (avec la méthode save)
# et leur attribuer 1 ou 0 comme suffixe selon si oui ou non on detecte un masque pour faciliter la récupération des données...
def resizeAllDatas():
    faceRec = FaceRecognition()
    for file in os.listdir(NOCONVERT_PATH):
        file_name, file_extension = os.path.splitext(file)
        image = cv2.imread(NOCONVERT_PATH+file_name+file_extension)
        _, resized = faceRec.auto_crop_image(image)
        if resized is not None:
            if len(resized) >= 1:
                for resizedOne in resized:
                    faceRec.save(resizedOne, 'found'+file_name)

#resizeAllDatas()
#trainMask = TrainMask()
#trainMask.train()
#del trainMask
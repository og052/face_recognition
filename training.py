import cv2
import numpy as np
from PIL import Image
import os


path2 = "data/"

recognizer = cv2.face.LBPHFaceRecognizer_create()  # Older syntax


def getImage():
    faces = []
    ids = []
    for x in os.listdir(path2):

        imagePath = [os.path.join(f"data/{x}/", f) for f in os.listdir(f"data/{x}/")]


        for i in imagePath:
            faceImage = Image.open(i).convert("L")
            faceNP = np.array(faceImage)
            id = (os.path.split(i)[-1].split(".")[1])
            id = int(id)
            faces.append(faceNP)
            ids.append(id)
            cv2.imshow("Training", faceNP)
            cv2.waitKey(1)

    return ids, faces




IDs, face_data = getImage()
recognizer.train(face_data, np.array(IDs))
recognizer.write("Trainer.yml")
cv2.destroyAllWindows()
print("Tranining completed............")
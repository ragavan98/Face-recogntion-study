import os
import cv2
import numpy as np
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create();
path='dataset'
def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceimg=Image.open(imagePath).convert('L');
        facenp=np.array(faceimg,'uint8')
        ID=int(os.path.split(imagePath)[1].split('-')[2])
        faces.append(facenp)
        print(ID)
        IDs.append(ID)
        cv2.imshow("training",facenp)
        cv2.waitKey(10)
    return IDs,faces

IDs,faces=getImagesWithID(path)
recognizer.train(faces,np.array(IDs))
recognizer.save('traineddata.yml')
cv2.destroyAllWindows()

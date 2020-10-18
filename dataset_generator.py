import cv2
import numpy as np
import os
#import mysql.connector
facedetect=cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml');

#"http://192.168.43.133:4747/video"
#cnx=mysql.connector.connect(user='root',password='',host='127.0.0.1',port='3306',database='attendance')
#cn=cnx.cursor()
id=input('Enter Student Name ')
rollno=str(id)
name=input('Enter Roll.no ')
#parentsno=int(input("Enter parents mobile.no "))
st=0
#cn.execute("INSERT INTO studentinfo VALUES(%s,%s,%s,0)",(rollno,name,parentsno))
#cnx.commit()
sampleNum=0;
cam=cv2.VideoCapture(0)
cv2.resizeWindow("Face",800,600);
while(True):
    ret,im=cam.read();
    #img=cv2.imread("im",0)
    img1=cv2.resize(im,(800,600))
    #cv2.imshow("face",img1);
    gray= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    
    faces=facedetect.detectMultiScale(gray,1.1,5);
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1;
        print(sampleNum)
        face=gray[y-3:y+h+7, x-3:x+w+7]
        cv2.imwrite("dataset/User-"+str(id)+"-"+str(name)+"-"+str(sampleNum)+".jpg",face)
        
        cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),2)
        
        if(sampleNum==210):
            break
    cv2.imshow("Face",img1);
    cv2.waitKey(1);
    if(sampleNum==210):
        break


        
    cv2.waitKey(1);
##os.system('trainer.py');
cam.release()
cv2.destroyAllWindows()

    

import cv2
#import mysql.connector
import numpy as np
import os
import datetime
import time as t
path='dataset'
now=datetime.datetime.now()
date=str(now.day)+'/'+str(now.month)+'/'+str(now.year)
#cnx=mysql.connector.connect(user='root',password='',host='127.0.0.1',port='3306',database='attendance')
#cn=cnx.cursor()



#cn.execute("UPDATE studentinfo SET status=(%s) WHERE status=(%s)",('0','1'))
faceDetect=cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml');
cam=cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 10.0, (640,480))
#"http://192.168.43.237:4747/video"
cv2.resizeWindow("face",1000,700);
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("traineddata.yml")
id=0
z=1
namelist=[" "]
imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
font=cv2.FONT_HERSHEY_COMPLEX_SMALL
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.1,5);
    known=0
    for(x,y,w,h)in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y-3:y+h+7,x-3:x+w+7])
        confi=str(int(conf))
        
        
        for imagePath in imagePaths:
            ID=int(os.path.split(imagePath)[1].split('-')[2])
            
            if(id==ID and conf<80):
                name=str(os.path.split(imagePath)[-1].split('-')[1])
                known=1
                if name in namelist:
                    cv2.rectangle(img,(x-5,y-5),(x+w+5,y+h+5),(0,222,255),1)
                    cv2.putText(img,str(name),(x,y),font,1,(255,255,255));
                    
                                        
                else:
                    namelist.append(name)
                    time=str(datetime.datetime.now().hour)+':'+str(datetime.datetime.now().minute)
                    print('@  '+name+'    Time-'+time+'     Date:'+date);
                    #cn.execute("INSERT INTO register VALUES(%s,%s,%s,%s)",(ID,name,date,time))
                    #cn.execute("UPDATE studentinfo SET status=(%s) WHERE Rollno=(%s)",('1',str(ID)))
                    #cnx.commit()
                                        
            else:
                
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                name='unknown'
                #cv2.putText(img,'unknown',(x,y),font,1,(255,255,255));
            
        if(known==0):
            #frame = cv2.flip(img,1)
            # write the flipped frame
            cv2.rectangle(img,(x-7,y-7),(x+w,y+h),(255,123,0),1)   
            #cv2.putText(img,'unknown  '+confi,(x-8,y-8),font,1,(255,255,255));
            out.write(img)
            
            
    
        
    img1=cv2.resize(img,(1000,700))
    #equ = cv2.equalizeHist(img1)
    #dst = cv2.fastNlMeansDenoisingColoredMulti(img,img1,2,5,10,10,7,21)
    cv2.imshow("face",img1);
    if(cv2.waitKey(1)==ord('q')):
        cam.release()
        out.release()
        break;
cam.release()

cv2.destroyAllWindows()


    

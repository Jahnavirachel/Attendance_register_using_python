import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'pictures'
images = []
classNames = []
mylist = os.listdir(path)
#print(mylist)
for cls in mylist:
    currentimg = cv2.imread(f'{path}/{cls}')
    images.append(currentimg)
    classNames.append(os.path.splitext(cls)[0])
#print(classNames)

def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markAttendance(name):
    with open('Register.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%D,%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')




encodelistknownfaces = findEncodings(images)
print("Encoding Completed")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodelistknownfaces,encodeFace)
        faceDis = face_recognition.face_distance(encodelistknownfaces,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] :
            name = classNames[matchIndex].upper()
            #print(name)
            x,y,w,h = faceLoc
            #center_coordinates = x + w // 2, y + h // 2
            #radius = w // 2
            cv2.circle(img, (250,250), 100, (0, 0, 100), 3)
            cv2.putText(img,name,(x+6,w-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

            markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)





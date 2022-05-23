import cv2

import face_recognition

imgv = face_recognition.load_image_file('pictures/v1.jfif')
imgv = cv2.cvtColor(imgv,cv2.COLOR_BGR2RGB)
imgvtest = face_recognition.load_image_file('pictures/v.jfif')
imgvtest = cv2.cvtColor(imgvtest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgv)[0]
encodev = face_recognition.face_encodings(imgv)[0]
cv2.circle(imgv,(300,360),250,(120,1,220),2)


faceloctest = face_recognition.face_locations(imgvtest)[0]
encodevtest = face_recognition.face_encodings(imgvtest)[0]
cv2.circle(imgvtest,(200,200),100,(120,1,220),2)

results = face_recognition.compare_faces([encodev],encodevtest)
facedist = face_recognition.face_distance([encodev],encodevtest)
print(results,facedist)
cv2.putText(imgvtest,f'{results}{round(facedist[0],2)}',(50,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)

cv2.imshow('v1',imgv)
cv2.imshow('v',imgvtest)
cv2.waitKey(0)

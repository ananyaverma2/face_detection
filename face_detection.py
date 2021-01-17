import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('face_detection/haarcascade_frontalface_default.xml')


# for image

ori_img = cv2.imread('mk.jpg')

img = cv2.resize(ori_img, (840,500)) 
grayed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(grayed_image)

for x, y, w, h in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(255), randrange(255), randrange(255)), 5)

cv2.imshow('face_detection', img)
cv2.waitKey()


#for webcam

webcam = cv2.VideoCapture(0)

while True:

    happening, frame = webcam.read()
    grayed_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayed_image)

    for x, y, w, h in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(255), randrange(255), randrange(255)), 5)

    cv2.imshow('face_detection', frame)
    key = cv2.waitKey(1)

    if key==113 or key==81:
        break

webcam.release()

print ("code completed, you did it...!")



"""
動画から顔検出まで
"""

import glob 
import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers,models

face_cascade = cv2.CascadeClassifier('/Users/e185725/practice/fer_practice/ex/opencv_master/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/e185725/practice/fer_practice/ex/opencv_master/data/haarcascades_cuda/haarcascade_eye.xml')

checkpoint_path = "Training_log/cp.ckpt"
checkpointdir = os.path.dirname(checkpoint_path)


###モデルの作成
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(48,48,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dropout(0.2))#後から追加
model.add(layers.Dense(7,activation="softmax"))

###モデルをコンパイル
model.compile(optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
                )

model.load_weights(checkpoint_path)
emotion = ["angry","disgust","fear","happy","sad","surprise","neutral"]

#-----------------------
cap = cv2.VideoCapture(0)

while True:
    is_ok,frame = cap.read()
    if not is_ok :break
    #frame = cv2.resize(frame,(1100,540))
    #frame2 = copy.copy(frame)


    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray,(15,15),0)
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)
    for (x,y,w,h) in faces:
        if (w < 150 or h < 150 and len(faces) == 1):
            continue
        
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        gray_img = gray[x:x+w, y:y+h]
        
        gray_img = cv2.resize(gray_img,dsize=(48,48))
        gray_img = gray_img.reshape((1,48,48,1))
        print(gray_img.shape)
        predictions = model.predict( gray_img )
        pred = [np.argmax(i) for i in predictions][0]
        #print(pred)
        cv2.putText(frame, emotion[pred], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        

        """
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        """
    cv2.imshow("video",frame)
    key=cv2.waitKey(1)&0xFF

    if key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


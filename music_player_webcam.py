import time
import cv2
import os,random
import subprocess
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')

size = 4
# We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
global text
cap = cv2.VideoCapture(0)  # Using default WebCam connected to the PC.
now = time.time()###For calculate seconds of video
future = now + 10  ####here is second of time which taken by emotion recognition system ,you can change it
while True:
    ret,im=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces_detected = classifier.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])
        emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
        text = emotions[max_index]
        font = cv2.FONT_HERSHEY_TRIPLEX

        if text == 'Angry':
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 25, 255), 7)
            cv2.putText(im, text, (x + h, y), font, 1, (0, 25,255), 2)

        if text == 'Disgust':
            cv2.rectangle(im, (x, y), (x + w, y + h), (0,260,0), 7)
            cv2.putText(im, text, (x + h, y), font, 1, (0,260,0), 2)

        if text == 'Fear':
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 255), 7)
            cv2.putText(im, text, (x + h, y), font, 1, (0, 255, 255), 2)

        if text == 'Sad':
            cv2.rectangle(im, (x, y), (x + w, y + h), (0,191,255), 7)
            cv2.putText(im, text, (x + h, y), font, 1, (0,191,255), 2)
            
        if text == 'Happy':
            cv2.rectangle(im, (x, y), (x + w, y + h), (0,191,255), 7)
            cv2.putText(im, text, (x + h, y), font, 1, (0,191,255), 2)
            
        if text == 'Surprise':
            cv2.rectangle(im, (x, y), (x + w, y + h), (0,191,255), 7)
            cv2.putText(im, text, (x + h, y), font, 1, (0,191,255), 2)
            
        if text == 'Neutral':
            cv2.rectangle(im, (x, y), (x + w, y + h), (0,191,255), 7)
            cv2.putText(im, text, (x + h, y), font, 1, (0,191,255), 2)

    # Show the image/
    cv2.imshow('Music player with Emotion recognition', im)
    key = cv2.waitKey(5)& 0xff
    if time.time() > future:##after 20second music will play
        try:
            cv2.destroyAllWindows()
            mp = 'C:/Program Files (x86)/Windows Media Player/wmplayer.exe'
            if text == 'Angry':
                randomfile = random.choice(os.listdir("C:/Users/Adarsh Singh/Desktop/Deep Learning/My Project/Emotion Detection/Songs/Angry/"))
                print('You are angry !!!! please calm down:) ,I will play song for you :' + randomfile)
                file = ('C:/Users/Adarsh Singh/Desktop/Deep Learning/My Project/Emotion Detection/Songs/Angry/' + randomfile)
                subprocess.call([mp, file])

            if text == 'Happy':
                randomfile = random.choice(os.listdir("C:/Users/Adarsh Singh/Desktop/Deep Learning/My Project/Emotion Detection/Songs/Happy/"))
                print('You are smiling :) ,I am playing a special song for you: ' + randomfile)
                file = ('C:/Users/Adarsh Singh/Desktop/Deep Learning/My Project/Emotion Detection/Songs/Happy/' + randomfile)
                subprocess.call([mp, file])

            if text == 'Fear':
                randomfile = random.choice(os.listdir("C:/Users/Adarsh Singh/Desktop/Deep Learning/My Project/Emotion Detection/Songs/Fear/"))
                print('You have fear of something ,I am playing a song for you: ' + randomfile)
                file = ('C:/Users/Adarsh Singh/Desktop/Deep Learning/My Project/Emotion Detection/Songs/Fear/' + randomfile)
                subprocess.call([mp, file])

            if text == 'Sad':
                randomfile = random.choice(os.listdir("C:/Users/Adarsh Singh/Desktop/Deep Learning/My Project/Emotion Detection/Songs/Sad/"))
                print('You are sad,dont worry:) ,I am playing a song for you: ' + randomfile)
                file = ('C:/Users/Adarsh Singh/Desktop/Deep Learning/My Project/Emotion Detection/Songs/Sad/' + randomfile)
                subprocess.call([mp, file])
                
            if text == 'Surprise':
                randomfile = random.choice(os.listdir("C:/Users/Adarsh Singh/Desktop/Deep Learning/My Project/Emotion Detection/Songs/Surprise/"))
                print('You are surprised, just calm down:) ,I am playing a song for you: ' + randomfile)
                file = ('C:/Users/Adarsh Singh/Desktop/Deep Learning/My Project/Emotion Detection/Songs/Surprise/' + randomfile)
                subprocess.call([mp, file])
                
            if text == 'Disgust':
                randomfile = random.choice(os.listdir("C:/Users/Adarsh Singh/Desktop/Deep Learning/My Project/Emotion Detection/Songs/Disgust/"))
                print('You are disgusted ,let me cool you down:) ,I am playing a song for you: ' + randomfile)
                file = ('C:/Users/Adarsh Singh/Desktop/Deep Learning/My Project/Emotion Detection/Songs/Disgust/' + randomfile)
                subprocess.call([mp, file])
                
            if text == 'Neutral':
                randomfile = random.choice(os.listdir("C:/Users/Adarsh Singh/Desktop/Deep Learning/My Project/Emotion Detection/Songs/Neutral/"))
                print('You seems to be normal:) ,I am playing a song for you: ' + randomfile)
                file = ('C:/Users/Adarsh Singh/Desktop/Deep Learning/My Project/Emotion Detection/Songs/Neutral/' + randomfile)
                subprocess.call([mp, file])
            break

        except :
            print('Please stay focus in Camera frame atleast 15 seconds & run again this program:)')
            break

    if key == 27:  # The Esc key
        break
# add the required library
import mediapipe as mp 
import cv2 
import numpy as np 
import uuid 
import os
import math
import pyautogui

# for drawing and hands detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# cap = cv2.VideoCapture(0) #Windows
# cap = cv2.VideoCapture(0,cv2.CAP_AVFOUNDATION) #MacOS
cap = cv2.VideoCapture(0) #MacOS

def dlist(x1,y1,x2,y2):
    return math.sqrt(math.pow(x1-x2,2)) + math.sqrt(math.pow(y1-y2,2))

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
    #ret
    #frame
        ret, frame = cap.read()
        
        #BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #change Color Combinations
        
        #Flip on horizontal
        image = cv2.flip(image,1)
        
        #Set flag
        image.flags.writeable = True
    
        #Detections
        results = hands.process(image)
    
        # Set flag to true
        image.flags.writeable = True
    
        #RGB to BGR
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #print(results)
        
        DISTANCE = 0.06
        #Rendering results
        if results.multi_hand_landmarks:
            for num,hand in enumerate(results.multi_hand_landmarks):
                distanceUP = dlist(hand.landmark[4].x,hand.landmark[4].y,hand.landmark[8].x,hand.landmark[8].y)
                distanceDOWN = dlist(hand.landmark[4].x,hand.landmark[4].y,hand.landmark[16].x,hand.landmark[16].y)
                up =  distanceUP < DISTANCE
                down = distanceDOWN < DISTANCE
                
                if (up == True):
                    print("scroll up")
                    pyautogui.scroll(1)
                
                elif (down == True):
                    print("scroll down")
                    pyautogui.scroll(-1)
                    
                mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2),
                                         )
        image = cv2.resize(image, None, fx=0.5 , fy=0.5, interpolation = cv2.INTER_AREA)
        cv2.imshow('Hand Tracking', image)
    
        if (cv2.waitKey(10) & 0xFF == ord('q')):
            break
        
cap.release()
cv2.destroyAllWindows()
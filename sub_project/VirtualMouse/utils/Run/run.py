import re
import cv2
import mediapipe as mp
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import analysis as A
import Draw as D

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def ChooseOption(image,point_x,point_y,function_list,block):
  height, width = image.shape[0], image.shape[1]
  
  height_ratio,width_ratio= int(height/7),int(width/block)

  COLOR = (216,168,74)
  TEXT_COLOR = (255,255,255)
  GAP = 50
  THICKNESS = 3

  opt_list = []
  A.FindCoordList(image,opt_list,block,height_ratio,width_ratio)
  D.DrawRectangle(image,COLOR,THICKNESS,opt_list,block)
  D.DetectInRange(image,COLOR,point_x,point_y,opt_list,block)
  D.PutText(image,opt_list,function_list,GAP,TEXT_COLOR)

def main(cap,function_list):
  with mp_hands.Hands(max_num_hands=1,model_complexity=0,min_detection_confidence=0.8,min_tracking_confidence=0.5) as hands:    
    while cap.isOpened():
      ret, frame = cap.read()
      
      if (ret == False):
        break

      # Recolor Feed
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False        
      image = cv2.flip(image,1)
      # Make Detections
      results = hands.process(image)

      image.flags.writeable = True   
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      # Export coordinates
      # try:
      try:
        if results.multi_hand_landmarks:
          image, point_x, point_y = A.ExtractFinger(image,results,15,(0,255,0))
          ChooseOption(image,point_x,point_y,function_list,len(function_list))

      except:
        pass
      
      cv2.imshow('Virtual Mouse', image)

      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()
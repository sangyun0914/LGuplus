from tkinter import RIGHT
import numpy as np
import math
import cv2
from pyparsing import col

MHI_DURATION = 30
DEFAULT_THRESHOLD = 32

def dlist(x1,y1,x2,y2):
  return math.sqrt(math.pow(x1-x2,2)+ math.pow(y1-y2,2))

def DrawSkeleton(image, landmark_pose):
  image_height, image_width, _ = image.shape 
  COLOR = (0,0,0)

  cv2.rectangle(image, (0,0), (image_width,image_height), COLOR, cv2.FILLED)

  RADIUS = 5
  # get infor about pos
  RIGHT_SHOULDER = landmark_pose[12]
  RIGHT_SHOULDER_X = int(RIGHT_SHOULDER.x * image_width)
  RIGHT_SHOULDER_Y = int(RIGHT_SHOULDER.y * image_height)
  
  LEFT_SHOULDER = landmark_pose[11]
  LEFT_SHOULDER_X = int(LEFT_SHOULDER.x * image_width)
  LEFT_SHOULDER_Y = int(LEFT_SHOULDER.y * image_height)

  CENTER_SHOULDER_X = int((RIGHT_SHOULDER_X+LEFT_SHOULDER_X)/2)
  CENTER_SHOULDER_Y = int((RIGHT_SHOULDER_Y+LEFT_SHOULDER_Y)/2)

  RIGHT_ELBOW = landmark_pose[14]
  RIGHT_ELBOW_X = int(RIGHT_ELBOW.x * image_width)
  RIGHT_ELBOW_Y = int(RIGHT_ELBOW.y * image_height)

  LEFT_ELBOW = landmark_pose[13]
  LEFT_ELBOW_X = int(LEFT_ELBOW.x * image_width)
  LEFT_ELBOW_Y = int(LEFT_ELBOW.y * image_height)

  RIGHT_WRIST = landmark_pose[16]
  RIGHT_WRIST_X = int(RIGHT_WRIST.x * image_width)
  RIGHT_WRIST_Y = int(RIGHT_WRIST.y * image_height)

  LEFT_WRIST = landmark_pose[15]
  LEFT_WRIST_X = int(LEFT_WRIST.x * image_width)
  LEFT_WRIST_Y = int(LEFT_WRIST.y * image_height)

  RIGHT_HIP = landmark_pose[24]
  RIGHT_HIP_X = int(RIGHT_HIP.x * image_width)
  RIGHT_HIP_Y = int(RIGHT_HIP.y * image_height)

  LEFT_HIP = landmark_pose[23]
  LEFT_HIP_X = int(LEFT_HIP.x * image_width)
  LEFT_HIP_Y = int(LEFT_HIP.y * image_height)

  CENTER_HIP_X = int((RIGHT_HIP_X+LEFT_HIP_X)/2)
  CENTER_HIP_Y = int((RIGHT_HIP_Y+LEFT_HIP_Y)/2) 

  RIGHT_KNEE = landmark_pose[26]
  RIGHT_KNEE_X = int(RIGHT_KNEE.x * image_width)
  RIGHT_KNEE_Y = int(RIGHT_KNEE.y * image_height)

  LEFT_KNEE = landmark_pose[25]
  LEFT_KNEE_X = int(LEFT_KNEE.x * image_width)
  LEFT_KNEE_Y = int(LEFT_KNEE.y * image_height)

  RIGHT_ANKEL = landmark_pose[28]
  RIGHT_ANKEL_X = int(RIGHT_ANKEL.x * image_width)
  RIGHT_ANKEL_Y = int(RIGHT_ANKEL.y * image_height)

  LEFT_ANKEL = landmark_pose[27]
  LEFT_ANKEL_X = int(LEFT_ANKEL.x * image_width)
  LEFT_ANKEL_Y = int(LEFT_ANKEL.y * image_height)

  cv2.circle(image, (RIGHT_SHOULDER_X,RIGHT_SHOULDER_Y), RADIUS , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
  cv2.circle(image, (LEFT_SHOULDER_X,LEFT_SHOULDER_Y), RADIUS , (0,255,255) , cv2.FILLED , cv2.LINE_AA)
  cv2.circle(image, (RIGHT_ELBOW_X,RIGHT_ELBOW_Y), RADIUS , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
  cv2.circle(image, (LEFT_ELBOW_X,LEFT_ELBOW_Y), RADIUS , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
  cv2.circle(image, (RIGHT_WRIST_X,RIGHT_WRIST_Y), RADIUS , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
  cv2.circle(image, (LEFT_WRIST_X,LEFT_WRIST_Y), RADIUS , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
  cv2.circle(image, (RIGHT_HIP_X,RIGHT_HIP_Y), RADIUS , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
  cv2.circle(image, (LEFT_HIP_X,LEFT_HIP_Y), RADIUS , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
  cv2.circle(image, (RIGHT_KNEE_X,RIGHT_KNEE_Y), RADIUS , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
  cv2.circle(image, (LEFT_KNEE_X,LEFT_KNEE_Y), RADIUS , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
  cv2.circle(image, (RIGHT_ANKEL_X,RIGHT_ANKEL_Y), RADIUS , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
  cv2.circle(image, (LEFT_ANKEL_X,LEFT_ANKEL_Y), RADIUS , (0,255,255) , cv2.FILLED , cv2.LINE_AA)

  cv2.line(image, (RIGHT_WRIST_X , RIGHT_WRIST_Y) , (RIGHT_ELBOW_X, RIGHT_ELBOW_Y), (0,255,255), 3, cv2.LINE_AA)
  cv2.line(image, (RIGHT_ELBOW_X , RIGHT_ELBOW_Y) , (RIGHT_SHOULDER_X, RIGHT_SHOULDER_Y), (0,255,255), 3, cv2.LINE_AA)
  cv2.line(image, (RIGHT_SHOULDER_X , RIGHT_SHOULDER_Y) , (LEFT_SHOULDER_X, LEFT_SHOULDER_Y), (0,255,255), 3, cv2.LINE_AA)
  cv2.line(image, (LEFT_SHOULDER_X , LEFT_SHOULDER_Y) , (LEFT_ELBOW_X, LEFT_ELBOW_Y), (0,255,255), 3, cv2.LINE_AA)
  cv2.line(image, (LEFT_ELBOW_X , LEFT_ELBOW_Y) , (LEFT_WRIST_X, LEFT_WRIST_Y), (0,255,255), 3, cv2.LINE_AA)
  cv2.line(image, (RIGHT_HIP_X , RIGHT_HIP_Y) , (RIGHT_KNEE_X, RIGHT_KNEE_Y), (0,255,255), 3, cv2.LINE_AA)
  cv2.line(image, (RIGHT_KNEE_X , RIGHT_KNEE_Y) , (RIGHT_ANKEL_X, RIGHT_ANKEL_Y), (0,255,255), 3, cv2.LINE_AA)
  cv2.line(image, (LEFT_HIP_X , LEFT_HIP_Y) , (LEFT_KNEE_X, LEFT_KNEE_X), (0,255,255), 3, cv2.LINE_AA)
  cv2.line(image, (LEFT_KNEE_X , LEFT_KNEE_Y) , (LEFT_ANKEL_X, LEFT_ANKEL_Y), (0,255,255), 3, cv2.LINE_AA)
  cv2.line(image, (RIGHT_HIP_X , RIGHT_HIP_Y) , (LEFT_HIP_X, LEFT_HIP_Y), (0,255,255), 3, cv2.LINE_AA)
  cv2.line(image, (CENTER_HIP_X , CENTER_HIP_Y) , (CENTER_SHOULDER_X, CENTER_SHOULDER_Y), (0,255,255), 3, cv2.LINE_AA)
  
  return image

def CalculateXMaxMinYMaxMin():
  pass

def main(VIDEO_PATH):

  # start detection
  cap = cv2.VideoCapture(VIDEO_PATH)
  cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
  cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 0)
  cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 0)
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      continue
    
    image_height, image_width, _ = image.shape

    cv2.imshow('auto focus off',image)
    # cv2.imshow('Skeleton MHI', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()

if __name__ == "__main__":
  VIDEO_PATH = 0
  main(VIDEO_PATH)
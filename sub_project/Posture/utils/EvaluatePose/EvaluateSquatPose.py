import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import csv
import pickle 
import pandas as pd
import warnings
import math

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# CONST
SQUAT, LUNGE, PUSHUP, NONE = 0,1,2,3
BAD, NORMAL, GOOD = 4,5,6

dictEval = {"4":"Bad", "5":"Normal", "6":"Good"}

def findAngle(x1, y1, x2, y2, cx, cy):
  division_degree_first = x2-cx
  if (division_degree_first <= 0):
    division_degree_first= 1

  division_degree_second = x1-cx
  if (division_degree_second <= 0):
    division_degree_second= 1

  theta = math.atan((y2-cy)/division_degree_first)-math.atan((y1-cy)/division_degree_second)
  degree = int(180/math.pi)*abs(theta)

  return degree

def EvalulateSquatPose(image,landmark_pose):
  image_height, image_width, _ = image.shape 

  RIGHT_HIP = landmark_pose[24]
  RIGHT_HIP_X = int(RIGHT_HIP.x * image_width)
  RIGHT_HIP_Y = int(RIGHT_HIP.y * image_height)
  if (RIGHT_HIP.visibility < 0.5):
    RIGHT_HIP_X = 1
    RIGHT_HIP_Y = 1
  
  LEFT_HIP = landmark_pose[23]
  LEFT_HIP_X = int(LEFT_HIP.x * image_width)
  LEFT_HIP_Y = int(LEFT_HIP.y * image_height)
  if (LEFT_HIP.visibility < 0.5):
    LEFT_HIP_X = 1
    LEFT_HIP_Y = 1

  RIGHT_ANKLE = landmark_pose[28]
  RIGHT_ANKLE_X = int(RIGHT_ANKLE.x * image_width)
  RIGHT_ANKLE_Y = int(RIGHT_ANKLE.y * image_height)
  if (RIGHT_ANKLE.visibility < 0.5):
    RIGHT_ANKLE_X = 1
    RIGHT_ANKLE_Y = 1
  
  LEFT_ANKLE = landmark_pose[27]
  LEFT_ANKLE_X = int(LEFT_ANKLE.x * image_width)
  LEFT_ANKLE_Y = int(LEFT_ANKLE.y * image_height)
  if (LEFT_ANKLE.visibility < 0.5):
    LEFT_ANKLE_X = 1
    LEFT_ANKLE_Y = 1

  RIGHT_KNEE = landmark_pose[26]
  RIGHT_KNEE_X = int(RIGHT_KNEE.x * image_width)
  RIGHT_KNEE_Y = int(RIGHT_KNEE.y * image_height)
  if (RIGHT_KNEE.visibility < 0.5):
    RIGHT_KNEE_X = 1
    RIGHT_KNEE_Y = 1
  
  LEFT_KNEE = landmark_pose[25]
  LEFT_KNEE_X = int(LEFT_KNEE.x * image_width)
  LEFT_KNEE_Y = int(LEFT_KNEE.y * image_height)
  if (LEFT_KNEE.visibility < 0.5):
    LEFT_KNEE_X = 1
    LEFT_KNEE_Y = 1
  
  degreeOfLeftLeg= int(findAngle(LEFT_ANKLE_X,LEFT_ANKLE_Y,LEFT_HIP_X,LEFT_HIP_Y,LEFT_KNEE_X,LEFT_KNEE_Y))
  degreeOfRightLeg = int(findAngle(RIGHT_ANKLE_X,RIGHT_ANKLE_Y,RIGHT_HIP_X,RIGHT_HIP_Y,RIGHT_KNEE_X,RIGHT_KNEE_Y))

  left = 0
  right = 0

  if (degreeOfLeftLeg >= 130):
    left = BAD
  elif (degreeOfLeftLeg<130 and degreeOfLeftLeg>=90):
    left = NORMAL
  else:
    left = GOOD

  if (degreeOfRightLeg >= 130):
    right = BAD
  elif (degreeOfRightLeg<130 and degreeOfRightLeg>=90):
    right = NORMAL
  else:
    right = GOOD

  cv2.putText(image, "{}".format(degreeOfLeftLeg) , (LEFT_KNEE_X-5,LEFT_KNEE_Y), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
  cv2.putText(image, "{}".format(degreeOfRightLeg) , (RIGHT_KNEE_X-5,RIGHT_KNEE_Y), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

  if (left < right):
    cv2.putText(image, "leg -> {} squat".format(str(dictEval[str(left)])) , (50,300), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
  else:
    cv2.putText(image, "leg -> {} squat".format(str(dictEval[str(right)])) , (50,300), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
from tkinter import RIGHT
import mediapipe as mp
import numpy as np
import math
import cv2

MHI_DURATION = 50
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

  return image

def CalculateXMaxMinYMaxMin():
  pass

# prepare for detection
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# start detection
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w = frame.shape[:2]
prev_frame = frame.copy()
motion_history = np.zeros((h, w), np.float32)
timestamp = 0

with mp_pose.Pose(min_detection_confidence=0.8,min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    
    image_height, image_width, _ = image.shape

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    image.flags.writeable = True
    if results.pose_landmarks:
      landmark_pose = results.pose_landmarks.landmark

      image = DrawSkeleton(image,landmark_pose)
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      frame_diff = cv2.absdiff(image, prev_frame)
      gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
      ret, fgmask = cv2.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
      timestamp += 1

      # update motion history
      cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)

      # normalize motion history
      # np.clip -> array 값이 지정한 최솟값보다 작으면 그  최솟값으로 지정, 만약 지정한 최댓값보다 크면 최댓값으로 고정
      # np.uint8 -> 1byte 만큼의 정수표현
      mh = np.uint8(np.clip((motion_history - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)

      im_color = cv2.applyColorMap(mh, cv2.COLORMAP_JET)

      cv2.imshow('motion-history', cv2.flip(im_color,1))
      
      prev_frame = frame.copy()
    # cv2.imshow('Skeleton MHI', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
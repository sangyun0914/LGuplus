from tkinter import RIGHT
import mediapipe as mp
import numpy as np
import math
import cv2

def dlist(x1,y1,x2,y2):
  return math.sqrt(math.pow(x1-x2,2)+ math.pow(y1-y2,2))

def IsStandOrLying(landmark_pose,image_height,image_width):
  RIGHT_SHOULDER = landmark_pose[12]
  LEFT_SHOULDER = landmark_pose[11]
  RIGHT_HIP = landmark_pose[24]
  LEFT_HIP = landmark_pose[23]
  RIGHT_KNEE = landmark_pose[26]
  LEFT_KNEE = landmark_pose[25]

  RIGHT_SHOULDER_TO_RIGHT_HIP = (int)(dlist(RIGHT_SHOULDER.x * image_width, RIGHT_SHOULDER.y * image_height, RIGHT_HIP.x * image_width, RIGHT_HIP.y * image_height))
  RIGHT_HIP_TO_RIGHT_KNEE = (int)(dlist(RIGHT_HIP.x * image_width,RIGHT_HIP.y * image_height,RIGHT_KNEE.x * image_width,RIGHT_KNEE.y * image_height))

  LEFT_SHOULDER_TO_LEFT_HIP = (int)(dlist(LEFT_SHOULDER.x * image_width,LEFT_SHOULDER.y * image_height,LEFT_HIP.x * image_width,LEFT_HIP.y * image_height))
  LEFT_HIP_TO_LEFT_KNEE = (int)(dlist(LEFT_HIP.x * image_width,LEFT_HIP.y * image_height,LEFT_KNEE.x * image_width,LEFT_KNEE.y * image_height))
  
  # if (RIGHT_SHOULDER.y - RIGHT_HIP.y > ):
  #   pass

def DrawSkeleton(image, landmark_pose):
  RIGHT_SHOULDER = landmark_pose[12]
  LEFT_SHOULDER = landmark_pose[11]
  RIGHT_ELBOW = landmark_pose[14]
  LEFT_ELBOW = landmark_pose[13]
  RIGHT_WRIST = landmark_pose[16]
  LEFT_WRIST = landmark_pose[15]
  RIGHT_HIP = landmark_pose[24]
  LEFT_HIP = landmark_pose[23]
  RIGHT_KNEE = landmark_pose[26]
  LEFT_KNEE = landmark_pose[25]
  RIGHT_ANKEL = landmark_pose[28]
  LEFT_ANKEL = landmark_pose[27]

# prepare for detection
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

state = "down"
count = -1

# start detection
cap = cv2.VideoCapture(0)

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
    
    SQUAT = 1
    STAND = 0
    pre_image = STAND
    cur_image = STAND

    # 24, 28
    # 23, 27

    image.flags.writeable = True
    if results.pose_landmarks:
      landmark_pose = results.pose_landmarks.landmark

      img = np.zeros((image_height, image_width,3), dtype=np.uint8)

      # # get infor about pos
      # RIGHT_SHOULDER = landmark_pose[12]
      # LEFT_SHOULDER = landmark_pose[11]

      # RIGHT_ELBOW = landmark_pose[14]
      # LEFT_ELBOW = landmark_pose[13]

      # RIGHT_WRIST = landmark_pose[16]
      # LEFT_WRIST = landmark_pose[15]

      # RIGHT_HIP = landmark_pose[24]
      # LEFT_HIP = landmark_pose[23]

      # RIGHT_KNEE = landmark_pose[26]
      # LEFT_KNEE = landmark_pose[25]

      # RIGHT_ANKEL = landmark_pose[28]
      # LEFT_ANKEL = landmark_pose[27]

      # cv2.circle(img, (RIGHT_SHOULDER.x , RIGHT_SHOULDER.y), 2 , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
      # cv2.circle(img, (LEFT_SHOULDER.x , LEFT_SHOULDER.y), 2 , (0,255,255) , cv2.FILLED , cv2.LINE_AA)
      # cv2.circle(img, (RIGHT_ELBOW.x , RIGHT_ELBOW.y), 2 , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
      # cv2.circle(img, (LEFT_ELBOW.x , LEFT_ELBOW.y), 2 , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
      # cv2.circle(img, (RIGHT_WRIST.x , RIGHT_WRIST.y), 2 , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
      # cv2.circle(img, (LEFT_WRIST.x , LEFT_WRIST.y), 2 , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
      # cv2.circle(img, (RIGHT_HIP.x , RIGHT_HIP.y), 2 , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
      # cv2.circle(img, (LEFT_HIP.x , LEFT_HIP.y), 2 , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
      # cv2.circle(img, (RIGHT_KNEE.x , RIGHT_KNEE.y), 2 , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
      # cv2.circle(img, (LEFT_KNEE.x , LEFT_KNEE.y), 2 , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
      # cv2.circle(img, (RIGHT_ANKEL.x, RIGHT_ANKEL.y), 2 , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 
      # cv2.circle(img, (LEFT_ANKEL.x , LEFT_ANKEL.y), 2 , (0,255,255) , cv2.FILLED , cv2.LINE_AA) 

      mp_drawing.draw_landmarks(
        img,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


      # cv2.line(image, (landmark_pose[26].x * image_width , landmark_pose[26].y * image_height) , (landmark_pose[28].x * image_width,landmark_pose[28].y * image_height), (0,255,0), 3, cv2.LINE_8)
      targetDistance = dlist(landmark_pose[26].x * image_width, landmark_pose[26].y * image_height, landmark_pose[28].x * image_width, landmark_pose[28].y * image_height)
      distanceRight = dlist(landmark_pose[24].x * image_width, landmark_pose[24].y * image_height, landmark_pose[28].x * image_width, landmark_pose[28].y * image_height)
      distanceLeft = dlist(landmark_pose[23].x * image_width, landmark_pose[23].y * image_height, landmark_pose[27].x * image_width, landmark_pose[27].y * image_height)
      targetDistance = int(targetDistance)
      targetDistance = targetDistance * 1.1

      distanceRight = int(distanceRight)
      distanceLeft = int(distanceLeft)

      if (distanceRight < targetDistance):
        cur_image = SQUAT
      
      elif (distanceLeft < targetDistance):
        cur_image = SQUAT

      else :
        cur_image = STAND

      if (cur_image == SQUAT):
        state = "down"

      if (cur_image == STAND and state == "down"):
        state = "up"
        count += 1
        print("Total squat : ", count)

      # print("Right : {}, Left : {}".format(distanceRight,distanceLeft))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    cv2.imshow('squat assist', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
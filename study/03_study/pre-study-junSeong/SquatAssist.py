import mediapipe as mp
import numpy as np
import math
import cv2

def dlist(x1,y1,x2,y2):
    return math.sqrt(math.pow(x1-x2,2)+ math.pow(y1-y2,2))

# prepare for detection
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

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
      if (cur_image == STAND):
        pre_image = STAND
      else:
        pre_image = SQUAT
        
      landmark_pose = results.pose_landmarks.landmark

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

      if (pre_image == SQUAT and cur_image == STAND):
        print("UP")

      elif (pre_image == STAND and cur_image == SQUAT):
        print("DOWN")

      print("pre : {} , cur : {}".format(pre_image, cur_image))
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
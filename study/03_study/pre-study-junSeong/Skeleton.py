from tkinter import RIGHT
import mediapipe as mp
import numpy as np
import math
import cv2

# prepare for detection
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

MHI_DURATION = 50
DEFAULT_THRESHOLD = 32

# start detection
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.8,min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    
    cv2.imshow("original", image)

    image_height, image_width, _ = image.shape

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # joint 정보를 얻어옴
    results = pose.process(image)

    image.flags.writeable = True
    if results.pose_landmarks:
      landmark_pose = results.pose_landmarks.landmark

      img = np.zeros((image_height, image_width,3), dtype=np.uint8)
      
      mp_drawing.draw_landmarks(
        img,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255,255,255), thickness=4, circle_radius=4),
        mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2),)

      cv2.imshow("black and skeleton",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
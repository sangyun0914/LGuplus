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
ret, frame = cap.read()
h, w = frame.shape[:2]
prev_frame = frame.copy()
motion_history = np.zeros((h, w), np.float32)
timestamp = 0

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
        mp_drawing.DrawingSpec(color=(0,0,0), thickness=4, circle_radius=4),
        mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2),)

      cv2.imshow("black and skeleton",img)

      frame_diff = cv2.absdiff(img, prev_frame)
      gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
      ret, fgmask = cv2.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
      timestamp += 1

      # update motion history
      cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)

      mh = np.uint8(np.clip((motion_history - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)
      cv2.imshow('motion-history', mh)

      prev_frame = mh.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
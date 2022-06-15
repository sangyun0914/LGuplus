import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

if not cap.isOpened():
  exit()

while cap.isOpened():
  #프레임 캡쳐
  ret, frame_original = cap.read()
  width, height , channel = frame_original.shape

  frame_blur = frame_original
  frame_yolo = frame_original
  frame_canny = frame_original
  frame_background = frame_original
  empty = np.zeros((width,height,channel), np.uint8)

  horizontal_stack1 = np.hstack((frame_original,frame_blur))
  horizontal_stack2 = np.hstack((frame_canny,frame_background))
  horizontal_stack3 = np.hstack((frame_yolo,empty))

  view = np.vstack((horizontal_stack1,horizontal_stack2,horizontal_stack3))

  cv2.imshow('Realtime-processing', view)

  if cv2.waitKey(1) == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
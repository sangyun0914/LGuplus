import numpy as np
import cv2

cap = cv2.VideoCapture('video/videoplayback.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
  ret, frame = cap.read()

  if (ret == False):
    break;

  fgmask = fgbg.apply(frame)

  
  cv2.imshow('fgmask',fgmask)
  cv2.imshow('frame', frame)

  if cv2.waitKey(1) == ord('q'):
    break;

cap.release()
cv2.destroyAllWindows()
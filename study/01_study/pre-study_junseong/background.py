import numpy as np
import cv2

video = cv2.VideoCapture(0)

while video.isOpened():
  ret, img = video.read()
  # Read image
  hh, ww = img.shape[:2]

# threshold on white
# Define lower and uppper limits
  lower = np.array([200, 200, 200])
  upper = np.array([255, 255, 255])

# Create mask to only select black
  thresh = cv2.inRange(img, lower, upper)

# apply morphology
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
  morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# invert morp image
  mask = 255 - morph

# apply mask to image
  result = cv2.bitwise_and(img, img, mask=mask)

  cv2.imshow('front',result)
  if cv2.waitKey(1) == ord('q'):
    break

video.release()
cv2.destroyAllWindows()
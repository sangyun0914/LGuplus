import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

BG_COLOR = (192, 192, 192) # gray
cap = cv2.VideoCapture(0)
with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1) as selfie_segmentation:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = selfie_segmentation.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    condition = np.stack(
      (results.segmentation_mask,) * 3, axis=-1) > 0.1

    # bg_image = cv2.imread('image.jpg')
    # bg_image=cv2.resize(bg_image, (1280, 720))
    bg_image = cv2.GaussianBlur(image,(55,55),0)
    output_image = np.where(condition, image, bg_image)

    cv2.imshow('MediaPipe Selfie Segmentation', output_image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
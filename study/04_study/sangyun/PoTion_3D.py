import cv2
from matplotlib.pyplot import annotate
import numpy as np
import mediapipe as mp
import pathlib
import collections
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

DIR = pathlib.Path(__file__).parent.resolve()

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
idx = 1
        
# For webcam input:
cap = cv2.VideoCapture(str(DIR) + "/video/squat.mp4")
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break
    
    cv2.namedWindow("original")
    cv2.namedWindow("joints")
    cv2.moveWindow("original", 1100,0)
    cv2.moveWindow("joints", 1400,0)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        cv2.imshow("original", image)
        
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            continue

        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        
        annotated_image = np.zeros((500,400,3), np.uint8)
        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow('joints', annotated_image)
        cv2.imwrite(str(DIR) + 'video/3D_Joints/annotated_image' + str(idx) + '.png', annotated_image)
        # Plot pose world landmarks.
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        idx += 1
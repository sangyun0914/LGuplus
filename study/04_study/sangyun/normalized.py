import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For webcam input:
cap = cv2.VideoCapture('study/04_study/sangyun/video/squat.mp4')
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image_height, image_width, _ = image.shape
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    world_landmark = results.pose_world_landmarks
    #print(type(world_landmark.landmark._values[00]))

    # FRAME, JOINTS, X, Y
    landmark_arr = []
    normalized_landmark_arr = []
    blank_image = np.zeros((image_height,image_width,3), np.uint8)

    for joint in range(33):
        landmark_arr.append((world_landmark.landmark._values[joint].x, world_landmark.landmark._values[joint].y))
        new_X = (world_landmark.landmark._values[joint].x + 1) / 2
        new_Y = (world_landmark.landmark._values[joint].y + 1) / 2
        normalized_landmark_arr.append((new_X, new_Y))
        cv2.circle(
            blank_image, 
            (int(new_X * image_width), 
            int(new_Y * image_height)), 
            3, 
            (0,0,255), 
            -1)

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.namedWindow("Normalized")
    cv2.moveWindow("Normalized",300,0)
    cv2.imshow('MediaPipe Pose', image)
    cv2.imshow("Normalized", blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cap.release()
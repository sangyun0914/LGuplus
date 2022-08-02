import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import multiprocessing as mulpro
import timeit

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) 

def extract_keypoints(results):
    allpose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten() if results.pose_world_landmarks else np.zeros(33*4)
    pose=allpose[44:68]
    pose=np.append(pose,allpose[92:116])
    return pose

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture("/Users/chaesiheon/Desktop/Code/LGuplus/study/04_study/siheon/mov.mov")
def test():
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()
            # Make detections
            image, results = mediapipe_detection(frame, pose)


            # NEW Export keypoints
            keypoints = extract_keypoints(results)

            # Draw landmarks
            draw_styled_landmarks(image, results)
            cv2.imshow('OpenCV Feed', image)
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

def test2():
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap2.isOpened():

            # Read feed
            ret2, frame2 = cap2.read()
            # Make detections
            image2, results2 = mediapipe_detection(frame2, pose)


            # NEW Export keypoints
            keypoints2 = extract_keypoints(results2)

            # Draw landmarks
            draw_styled_landmarks(image2, results2)
            cv2.imshow('OpenCV Feed', image2)
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
if __name__ == '__main__':
    #멀티 프로세싱
    th1 = mulpro.Process(target=test, args=())
    th2 = mulpro.Process(target=test2, args=())

    th1.start()
    th2.start()
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

import tensorflow as tf

model = tf.keras.models.load_model('ActionV1.h5')

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten() if results.pose_world_landmarks else np.zeros(33*4)

    return pose

# 1. New detection variables
sequence = []
threshold = 0.8

# Actions that we try to detect
actions = np.array(['lunge','lying','pushup','squat','stand'])

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("videoTest/202207201402_original.avi")

# Set mediapipe model 
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, pose)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-5:]
        
        if len(sequence) == 5:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            # print(actions[np.argmax(res)])
            action = actions[np.argmax(res)]
            # predictions.append(np.argmax(res))
            
        # #3. Viz logic
        #     if np.unique(predictions[-10:])[0]==np.argmax(res): 
        #         if res[np.argmax(res)] > threshold: 
                    
        #             if len(sentence) > 0: 
        #                 if actions[np.argmax(res)] != sentence[-1]:
        #                     sentence.append(actions[np.argmax(res)])
        #             else:
        #                 sentence.append(actions[np.argmax(res)])

        #     if len(sentence) > 5: 
        #         sentence = sentence[-5:]

        #     Viz probabilities
        #     image = prob_viz(res, actions, image, colors)
            print(action)
            # cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, action, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
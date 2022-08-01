import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import TensorBoard
import timeit

import tensorflow as tf

model = tf.keras.models.load_model('action.h5')


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

colors = [(245,117,16), (117,245,16), (16,117,245),(16,245,117),(117,16,245),(245,16,117),(167,213,8)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*200), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.95

# Actions that we try to detect
actions = np.array(['squat-down','squat-up','pushup-down','pushup-up','lunge-down','lunge-up','stand'])

cap = cv2.VideoCapture("/Users/chaesiheon/Downloads/Lunge, Alternating.mp4")
window=[]
squatcnt=0
pushupcnt=0
lungecnt=0
# Set mediapipe model 
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    i=0
    sum=0
    SFPS=""
    while cap.isOpened():
        start_t = timeit.default_timer()
        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, pose)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        window.append(keypoints)
        if len(window)==30:
            res = model.predict(np.expand_dims(window, axis=0))[0]
            # print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
                
                
            #3. Viz logic

            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)]=="squat-up" and sentence[-1]=="squat-down":
                        sentence.append(actions[np.argmax(res)])
                        squatcnt+=1
                    elif actions[np.argmax(res)]=="pushup-up" and sentence[-1]=="pushup-down":
                        sentence.append(actions[np.argmax(res)])
                        pushupcnt+=1
                    elif actions[np.argmax(res)]=="lunge-up" and sentence[-1]=="lunge-down":
                        sentence.append(actions[np.argmax(res)])
                        lungecnt+=1
                    else:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 1: 
                sentence = sentence[-1:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (500, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image,"squatcnt "+str(squatcnt),(3,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
            cv2.putText(image,"pushupcnt "+str(pushupcnt),(3,550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
            cv2.putText(image,"lungecnt "+str(lungecnt),(3,600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
            # Show to screen
            terminate_t = timeit.default_timer()
            FPS = 1./(terminate_t - start_t )
            sum=FPS+sum
            if SFPS=="":
                SFPS=str(FPS)
            if i==10:
                sum=sum/10
                sum = round(sum,4)
                SFPS=str(sum)
                i=0
                sum=0
            cv2.putText(image, "FPS : "+SFPS,(640, 60), 0, 1, (255,0,0),3)
            cv2.imshow('OpenCV Feed', image)
            i+=1
            window=window[-29:]
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
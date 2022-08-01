import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import csv
import pickle 
import pandas as pd
import warnings
import math
import sys
import os
from ..EvaluatePose import EvaluateSquatPose as esp

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# CONST
SQUAT, LUNGE, PUSHUP, NONE = 0,1,2,3
BAD, NORMAL, GOOD = 4,5,6

def ActionPerformed(prev,cur):
  if (prev == "squat" and cur == "stand"):
    return SQUAT
  
  elif (prev == "lunge" and cur == "stand"):
    return LUNGE

  elif (prev == "pushup" and cur == "lying"):
    return PUSHUP

  else:
    return NONE

def InferenceEngine(cap,MODEL):
  NumSquat,NumLunge,NumPushup = 0,0,0

  with open(MODEL, 'rb') as f:
    model = pickle.load(f)
  
  # Initiate holistic model
  with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5) as pose:
      prev = "stand"
      cur = "stand"

      while cap.isOpened():
          ret, frame = cap.read()
          
          if (ret == False):
            break

          # Recolor Feed
          image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          image.flags.writeable = False        
          
          # Make Detections
          results = pose.process(image)
          # print(results.face_landmarks)
          
          # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
          
          # Recolor image back to BGR for rendering
          image.flags.writeable = True   
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

          mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

          # Export coordinates
          # try:
          if results.pose_world_landmarks:
            row = list(np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten())
            # Append class name

            # Make Detections
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            
            # # Posture Recognition
            # if (body_language_class == "stand"):
            #   with mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
            #     pass

            cur = body_language_class

            if (cur == "squat"):
              esp(image,results.pose_landmarks.landmark)
            elif (cur == "lunge"):
              pass

            elif (cur == "pushup"):
              pass

            doAction = ActionPerformed(prev,cur)

            if (doAction == SQUAT):
              NumSquat += 1

            elif (doAction == LUNGE):
              NumLunge += 1

            elif (doAction == PUSHUP):
              NumPushup += 1

            # Get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            # Display Class
            cv2.putText(image, 'action'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'prob'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, "Squat: {}".format(str(NumSquat)) , (50,150), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
            cv2.putText(image, "Lunge: {}".format(str(NumLunge)) , (50,200), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
            cv2.putText(image, "Pushup: {}".format(str(NumPushup)) , (50,250), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

            prev = cur

          cv2.imshow('Skeleton Action Classifier', image)

          if cv2.waitKey(10) & 0xFF == ord('q'):
              break

  cap.release()
  cv2.destroyAllWindows()
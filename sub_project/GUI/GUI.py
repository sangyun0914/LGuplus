# Import required Libraries
from argparse import Action
from telnetlib import STATUS
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import mediapipe as mp # Import mediapipe
import numpy as np
import pickle 
import pandas as pd
import math

# https://m.blog.naver.com/chandong83/221124467992

# Draw mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

prev = "stand"
cur = "stand"
NumSquat = 0
NumLunge = 0
NumPushup = 0

def findAngle(x1, y1, x2, y2, cx, cy):
  division_degree_first = x2-cx
  if (division_degree_first <= 0):
    division_degree_first= 1

  division_degree_second = x1-cx
  if (division_degree_second <= 0):
    division_degree_second= 1

  theta = math.atan((y2-cy)/division_degree_first)-math.atan((y1-cy)/division_degree_second)
  degree = int(180/math.pi)*abs(theta)

  return degree

def EvalulateSquatPose(image,landmark_pose):
  image_height, image_width, _ = image.shape 

  # # 좌표를 얻어옴
  # RIGHT_SHOULDER = landmark_pose[12]
  # RIGHT_SHOULDER_X = int(RIGHT_SHOULDER.x * image_width)
  # RIGHT_SHOULDER_Y = int(RIGHT_SHOULDER.y * image_height)
  # if (RIGHT_SHOULDER.visibility < 0.5):
  #   RIGHT_SHOULDER_X = 1
  #   RIGHT_SHOULDER_Y = 1

  # LEFT_SHOULDER = landmark_pose[11]
  # LEFT_SHOULDER_X = int(LEFT_SHOULDER.x * image_width)
  # LEFT_SHOULDER_Y = int(LEFT_SHOULDER.y * image_height)
  # if (LEFT_SHOULDER.visibility < 0.5):
  #   LEFT_SHOULDER_X = 1
  #   LEFT_SHOULDER_Y = 1

  # RIGHT_WRIST = landmark_pose[16]
  # RIGHT_WRIST_X = int(RIGHT_WRIST.x * image_width)
  # RIGHT_WRIST_Y = int(RIGHT_WRIST.y * image_height)
  # if (RIGHT_WRIST.visibility < 0.5):
  #   RIGHT_WRIST_X = 1
  #   RIGHT_WRIST_Y = 1

  # LEFT_WRIST = landmark_pose[15]
  # LEFT_WRIST_X = int(LEFT_WRIST.x * image_width)
  # LEFT_WRIST_Y = int(LEFT_WRIST.y * image_height)
  # if (LEFT_WRIST.visibility < 0.5):
  #   LEFT_WRIST_X = 1
  #   LEFT_WRIST_X = 1
  
  # LEFT_ELBOW = landmark_pose[13]
  # LEFT_ELBOW_X = int(LEFT_ELBOW.x * image_width)
  # LEFT_ELBOW_Y = int(LEFT_ELBOW.y * image_height)
  # if (LEFT_ELBOW.visibility < 0.5):
  #   LEFT_ELBOW_X = 1
  #   LEFT_ELBOW_Y = 1

  # RIGHT_ELBOW = landmark_pose[14]
  # RIGHT_ELBOW_X = int(RIGHT_ELBOW.x * image_width)
  # RIGHT_ELBOW_Y = int(RIGHT_ELBOW.y * image_height)
  # if (RIGHT_ELBOW.visibility < 0.5):
  #   RIGHT_ELBOW_X = 1
  #   RIGHT_ELBOW_Y = 1

  RIGHT_HIP = landmark_pose[24]
  RIGHT_HIP_X = int(RIGHT_HIP.x * image_width)
  RIGHT_HIP_Y = int(RIGHT_HIP.y * image_height)
  if (RIGHT_HIP.visibility < 0.5):
    RIGHT_HIP_X = 1
    RIGHT_HIP_Y = 1
  
  LEFT_HIP = landmark_pose[23]
  LEFT_HIP_X = int(LEFT_HIP.x * image_width)
  LEFT_HIP_Y = int(LEFT_HIP.y * image_height)
  if (LEFT_HIP.visibility < 0.5):
    LEFT_HIP_X = 1
    LEFT_HIP_Y = 1

  RIGHT_ANKLE = landmark_pose[28]
  RIGHT_ANKLE_X = int(RIGHT_ANKLE.x * image_width)
  RIGHT_ANKLE_Y = int(RIGHT_ANKLE.y * image_height)
  if (RIGHT_ANKLE.visibility < 0.5):
    RIGHT_ANKLE_X = 1
    RIGHT_ANKLE_Y = 1
  
  LEFT_ANKLE = landmark_pose[27]
  LEFT_ANKLE_X = int(LEFT_ANKLE.x * image_width)
  LEFT_ANKLE_Y = int(LEFT_ANKLE.y * image_height)
  if (LEFT_ANKLE.visibility < 0.5):
    LEFT_ANKLE_X = 1
    LEFT_ANKLE_Y = 1

  RIGHT_KNEE = landmark_pose[26]
  RIGHT_KNEE_X = int(RIGHT_KNEE.x * image_width)
  RIGHT_KNEE_Y = int(RIGHT_KNEE.y * image_height)
  if (RIGHT_KNEE.visibility < 0.5):
    RIGHT_KNEE_X = 1
    RIGHT_KNEE_Y = 1
  
  LEFT_KNEE = landmark_pose[25]
  LEFT_KNEE_X = int(LEFT_KNEE.x * image_width)
  LEFT_KNEE_Y = int(LEFT_KNEE.y * image_height)
  if (LEFT_KNEE.visibility < 0.5):
    LEFT_KNEE_X = 1
    LEFT_KNEE_Y = 1
  
  degreeOfLeftLeg= int(findAngle(LEFT_ANKLE_X,LEFT_ANKLE_Y,LEFT_HIP_X,LEFT_HIP_Y,LEFT_KNEE_X,LEFT_KNEE_Y))
  degreeOfRightLeg = int(findAngle(RIGHT_ANKLE_X,RIGHT_ANKLE_Y,RIGHT_HIP_X,RIGHT_HIP_Y,RIGHT_KNEE_X,RIGHT_KNEE_Y))
  # degreeOfLeftArm = int(findAngle(LEFT_WRIST_X,LEFT_WRIST_Y,LEFT_SHOULDER_X,LEFT_SHOULDER_Y,LEFT_ELBOW_X,LEFT_ELBOW_Y))
  # degreeOfRightArm = int(findAngle(RIGHT_WRIST_X,RIGHT_WRIST_Y,RIGHT_SHOULDER_X,RIGHT_SHOULDER_Y,RIGHT_ELBOW_X,RIGHT_ELBOW_Y))

  resultOfSquat_left = 0
  resultOfSquat_right = 0

  if (degreeOfLeftLeg >= 130):
    resultOfSquat_left = BAD
  elif (degreeOfLeftLeg<130 and degreeOfLeftLeg>=90):
    resultOfSquat_left = NORMAL
  else:
    resultOfSquat_left = GOOD

  if (degreeOfRightLeg >= 130):
    resultOfSquat_right = BAD
  elif (degreeOfRightLeg<130 and degreeOfRightLeg>=90):
    resultOfSquat_right = NORMAL
  else:
    resultOfSquat_right = GOOD

  return resultOfSquat_left,resultOfSquat_right

def ActionPerformed(prev,cur):
  if (prev == "squat" and cur == "stand"):
    return SQUAT
  
  elif (prev == "lunge" and cur == "stand"):
    return LUNGE

  elif (prev == "pushup" and cur == "lying"):
    return PUSHUP

  else:
    return NONE

def EvalDetect(image,landmark_pose):
  image_height, image_width, _ = image.shape 
  detection = True

  RIGHT_SHOULDER = landmark_pose[12]
  RIGHT_SHOULDER_X = int(RIGHT_SHOULDER.x * image_width)
  RIGHT_SHOULDER_Y = int(RIGHT_SHOULDER.y * image_height)
  if (RIGHT_SHOULDER.visibility < 0.5):
    detection = False

  LEFT_SHOULDER = landmark_pose[11]
  LEFT_SHOULDER_X = int(LEFT_SHOULDER.x * image_width)
  LEFT_SHOULDER_Y = int(LEFT_SHOULDER.y * image_height)
  if (LEFT_SHOULDER.visibility < 0.5):
    detection = False

  CENTER_SHOULDER_X = int((RIGHT_SHOULDER_X+LEFT_SHOULDER_X)/2)
  CENTER_SHOULDER_Y = int((RIGHT_SHOULDER_Y+LEFT_SHOULDER_Y)/2) 

  RIGHT_ELBOW = landmark_pose[14]
  RIGHT_ELBOW_X = int(RIGHT_ELBOW.x * image_width)
  RIGHT_ELBOW_Y = int(RIGHT_ELBOW.y * image_height)
  if (RIGHT_ELBOW.visibility < 0.5):
    detection = False

  LEFT_ELBOW = landmark_pose[13]
  LEFT_ELBOW_X = int(LEFT_ELBOW.x * image_width)
  LEFT_ELBOW_Y = int(LEFT_ELBOW.y * image_height)
  if (LEFT_ELBOW.visibility < 0.5):
    detection = False

  RIGHT_WRIST = landmark_pose[16]
  RIGHT_WRIST_X = int(RIGHT_WRIST.x * image_width)
  RIGHT_WRIST_Y = int(RIGHT_WRIST.y * image_height)
  if (RIGHT_WRIST.visibility < 0.5):
    detection = False

  LEFT_WRIST = landmark_pose[15]
  LEFT_WRIST_X = int(LEFT_WRIST.x * image_width)
  LEFT_WRIST_Y = int(LEFT_WRIST.y * image_height)
  if (LEFT_WRIST.visibility < 0.5):
    detection = False
  
  RIGHT_HIP = landmark_pose[24]
  RIGHT_HIP_X = int(RIGHT_HIP.x * image_width)
  RIGHT_HIP_Y = int(RIGHT_HIP.y * image_height)
  if (RIGHT_HIP.visibility < 0.5):
    detection = False
  
  LEFT_HIP = landmark_pose[23]
  LEFT_HIP_X = int(LEFT_HIP.x * image_width)
  LEFT_HIP_Y = int(LEFT_HIP.y * image_height)
  if (LEFT_HIP.visibility < 0.5):
    detection = False

  RIGHT_KNEE = landmark_pose[26]
  RIGHT_KNEE_X = int(RIGHT_KNEE.x * image_width)
  RIGHT_KNEE_Y = int(RIGHT_KNEE.y * image_height)
  if (RIGHT_KNEE.visibility < 0.5):
    detection = False
  
  LEFT_KNEE = landmark_pose[25]
  LEFT_KNEE_X = int(LEFT_KNEE.x * image_width)
  LEFT_KNEE_Y = int(LEFT_KNEE.y * image_height)
  if (LEFT_KNEE.visibility < 0.5):
    detection = False

  RIGHT_ANKLE = landmark_pose[28]
  RIGHT_ANKLE_X = int(RIGHT_ANKLE.x * image_width)
  RIGHT_ANKLE_Y = int(RIGHT_ANKLE.y * image_height)
  if (RIGHT_ANKLE.visibility < 0.5):
    detection = False
  
  LEFT_ANKLE = landmark_pose[27]
  LEFT_ANKLE_X = int(LEFT_ANKLE.x * image_width)
  LEFT_ANKLE_Y = int(LEFT_ANKLE.y * image_height)
  if (LEFT_ANKLE.visibility < 0.5):
    detection = False

  return detection

# Define function to show frame
def show_frames():
  global NumSquat,NumLunge,NumPushup,cur,prev

  with mp_pose.Pose(min_detection_confidence=0.8,min_tracking_confidence=0.5) as pose:
    # Get the latest frame and convert into Image
    src= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)

    # status
    img_status = np.zeros((int(height/2),WIDTH_INFOR,3) , dtype=np.uint8)
    STATUS_COLOR = (255,255,255)
    cv2.putText(img_status, 'squat ', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, STATUS_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img_status, '-> {}'.format(NumSquat), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, STATUS_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img_status, 'lunge', (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, STATUS_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img_status, '-> {}'.format(NumLunge), (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, STATUS_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img_status, 'pushup', (10,190), cv2.FONT_HERSHEY_SIMPLEX, 1, STATUS_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img_status, '-> {}'.format(NumPushup), (10,230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, STATUS_COLOR, 1, cv2.LINE_AA)
    img_status = Image.fromarray(img_status)

    imgtk_status = ImageTk.PhotoImage(image = img_status)
    status.imgtk = imgtk_status
    status.configure(image = imgtk_status)

    # webcam
    # Recolor Feed
    image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
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

    action = "None"

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

      action = body_language_class

      cur = action

      dictEval = {"4":"Bad", "5":"Normal", "6":"Good"}

      if (cur == "squat"):
        left, right = EvalulateSquatPose(image,results.pose_landmarks.landmark)
        if (left < right):
          cv2.putText(image, "left leg -> {} squat".format(str(dictEval[str(left)])) , (50,300), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
        else:
          cv2.putText(image, "right leg -> {} squat".format(str(dictEval[str(right)])) , (50,300), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

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
      image = cv2.resize(image,None,fx=0.5,fy=0.5)
      # cv2.putText(image, "Squat: {}".format(str(NumSquat)) , (50,150), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
      # cv2.putText(image, "Lunge: {}".format(str(NumLunge)) , (50,200), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
      # cv2.putText(image, "Pushup: {}".format(str(NumPushup)) , (50,250), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
    else:
      image = cv2.resize(image,None,fx=0.5,fy=0.5)
    img_cam = Image.fromarray(image)

    imgtk_cam = ImageTk.PhotoImage(image = img_cam)
    webcam.imgtk = imgtk_cam
    webcam.configure(image=imgtk_cam)

    # feedback
    img_feedback = np.zeros((int(height/2),WIDTH_INFOR,3) , dtype=np.uint8)
    cv2.putText(img_feedback, 'Action', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, STATUS_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img_feedback, '-> {}'.format(action), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, STATUS_COLOR, 1, cv2.LINE_AA)
    
    cv2.putText(img_feedback, 'Eval', (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, STATUS_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img_feedback, '->', (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, STATUS_COLOR, 1, cv2.LINE_AA)

    img_feedback = Image.fromarray(img_feedback)

    imgtk_feedback = ImageTk.PhotoImage(image = img_feedback)
    feedback.imgtk = imgtk_feedback
    feedback.configure(image = imgtk_feedback)

    prev = cur
    # Repeat after an interval to capture continiously
    webcam.after(5, show_frames)

# import model
MODEL = "ActionV6_rf.pkl"
# CONST
SQUAT, LUNGE, PUSHUP, NONE = 0,1,2,3
BAD, NORMAL, GOOD = 4,5,6
# Create an instance of TKinter Window or frame
win = Tk()

cap= cv2.VideoCapture(0)
# cap = cv2.VideoCapture("video/test.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 
WIDTH_INFOR = 180
GEOMETRY = str(int(width/2)+2*int(WIDTH_INFOR))+"x"+str(int(height/2))

# Set the size of the window
win.geometry(GEOMETRY)

# status label
status = Label(win)
status.grid(row=0, column=0)

# Create a Label to capture the Video frames
webcam = Label(win)
webcam.grid(row=0, column=1)

# feedback label
feedback = Label(win)
feedback.grid(row=0,column=2)

with open(MODEL, 'rb') as f:
    model = pickle.load(f)

show_frames()
win.mainloop()
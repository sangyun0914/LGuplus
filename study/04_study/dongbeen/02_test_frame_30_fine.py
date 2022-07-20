import os
from PIL import Image

import cv2
from cv2 import waitKey
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, tensor
from torchvision import transforms
import mediapipe as mp
import numpy as np
import math

MHI_DURATION = 30

LIST_COORD_RIGHT_SHOULDER = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_SHOULDER = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_CENTER_SHOULDER = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_RIGHT_ELBOW = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_ELBOW = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_RIGHT_WRIST = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_WRIST = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_RIGHT_HIP = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_HIP = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_CENTER_HIP = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_RIGHT_KNEE =[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_KNEE = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_RIGHT_ANKLE = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_ANKLE = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]

# 30개의 프레임 색깔 넣기
COLOR = []
RADIUS = []
for idx in range(1,MHI_DURATION+1):
  ratio = (idx-1)/(MHI_DURATION-1)
  result = 5*ratio
  result += 3
  RADIUS.append(int(result))
RADIUS.reverse()

MHI_DURATION_FIRST = 15
for result_idx in range(1,MHI_DURATION_FIRST+1):
    ratio = (result_idx-1)/(MHI_DURATION_FIRST-1)
    COLOR.append((0,int(255*ratio), int(255*(1-ratio))))

MHI_DURATION_SECOND = 15
for result_idx in range(1,MHI_DURATION_SECOND+1):
    ratio = (result_idx-1)/(MHI_DURATION_SECOND-1)
    COLOR.append((int(255*ratio), int(255*(1-ratio)), 0))

print(COLOR)
print(RADIUS)
# list에 좌표 넣기
def InsertCoordinate(image,landmark_pose):
  image_height, image_width, _ = image.shape 
  cv2.rectangle(image, (0,0), (image_width,image_height), (0,0,0), cv2.FILLED)

  # 좌표를 얻어옴
  RIGHT_SHOULDER = landmark_pose[12]
  RIGHT_SHOULDER_X = int(RIGHT_SHOULDER.x * image_width)
  RIGHT_SHOULDER_Y = int(RIGHT_SHOULDER.y * image_height)
  if (RIGHT_SHOULDER.visibility < 0.5):
    RIGHT_SHOULDER_X = 0
    RIGHT_SHOULDER_Y = 0
  LIST_COORD_RIGHT_SHOULDER.insert(0,(RIGHT_SHOULDER_X,RIGHT_SHOULDER_Y))
  LIST_COORD_RIGHT_SHOULDER.pop(29)

  LEFT_SHOULDER = landmark_pose[11]
  LEFT_SHOULDER_X = int(LEFT_SHOULDER.x * image_width)
  LEFT_SHOULDER_Y = int(LEFT_SHOULDER.y * image_height)
  if (LEFT_SHOULDER.visibility < 0.5):
    LEFT_SHOULDER_X = 0
    LEFT_SHOULDER_Y = 0
  LIST_COORD_LEFT_SHOULDER.insert(0,(LEFT_SHOULDER_X,LEFT_SHOULDER_Y))
  LIST_COORD_LEFT_SHOULDER.pop(29)
  
  CENTER_SHOULDER_X = int((RIGHT_SHOULDER_X+LEFT_SHOULDER_X)/2)
  CENTER_SHOULDER_Y = int((RIGHT_SHOULDER_Y+LEFT_SHOULDER_Y)/2)
  LIST_COORD_CENTER_SHOULDER.insert(0,(CENTER_SHOULDER_X,CENTER_SHOULDER_Y))
  LIST_COORD_CENTER_SHOULDER.pop(29)  

  RIGHT_ELBOW = landmark_pose[14]
  RIGHT_ELBOW_X = int(RIGHT_ELBOW.x * image_width)
  RIGHT_ELBOW_Y = int(RIGHT_ELBOW.y * image_height)
  if (RIGHT_ELBOW.visibility < 0.5):
    RIGHT_ELBOW_X = 0
    RIGHT_ELBOW_Y = 0
  LIST_COORD_RIGHT_ELBOW.insert(0,(RIGHT_ELBOW_X,RIGHT_ELBOW_Y))
  LIST_COORD_RIGHT_ELBOW.pop(29)

  LEFT_ELBOW = landmark_pose[13]
  LEFT_ELBOW_X = int(LEFT_ELBOW.x * image_width)
  LEFT_ELBOW_Y = int(LEFT_ELBOW.y * image_height)
  if (LEFT_ELBOW.visibility < 0.5):
    LEFT_ELBOW_X = 0
    LEFT_ELBOW_Y = 0
  LIST_COORD_LEFT_ELBOW.insert(0,(LEFT_ELBOW_X,LEFT_ELBOW_Y))
  LIST_COORD_LEFT_ELBOW.pop(29)

  RIGHT_WRIST = landmark_pose[16]
  RIGHT_WRIST_X = int(RIGHT_WRIST.x * image_width)
  RIGHT_WRIST_Y = int(RIGHT_WRIST.y * image_height)
  if (RIGHT_WRIST.visibility < 0.5):
    RIGHT_WRIST_X = 0
    RIGHT_WRIST_Y = 0
  LIST_COORD_RIGHT_WRIST.insert(0,(RIGHT_WRIST_X,RIGHT_WRIST_Y))
  LIST_COORD_RIGHT_WRIST.pop(29)

  LEFT_WRIST = landmark_pose[15]
  LEFT_WRIST_X = int(LEFT_WRIST.x * image_width)
  LEFT_WRIST_Y = int(LEFT_WRIST.y * image_height)
  if (LEFT_WRIST.visibility < 0.5):
    LEFT_WRIST_X = 0
    LEFT_WRIST_X = 0
  LIST_COORD_LEFT_WRIST.insert(0,(LEFT_WRIST_X,LEFT_WRIST_Y))
  LIST_COORD_LEFT_WRIST.pop(29)
  
  RIGHT_HIP = landmark_pose[24]
  RIGHT_HIP_X = int(RIGHT_HIP.x * image_width)
  RIGHT_HIP_Y = int(RIGHT_HIP.y * image_height)
  if (RIGHT_HIP.visibility < 0.5):
    RIGHT_HIP_X = 0
    RIGHT_HIP_Y = 0
  LIST_COORD_RIGHT_HIP.insert(0,(RIGHT_HIP_X,RIGHT_HIP_Y))
  LIST_COORD_RIGHT_HIP.pop(29)
  
  LEFT_HIP = landmark_pose[23]
  LEFT_HIP_X = int(LEFT_HIP.x * image_width)
  LEFT_HIP_Y = int(LEFT_HIP.y * image_height)
  if (LEFT_HIP.visibility < 0.5):
    LEFT_HIP_X = 0
    LEFT_HIP_Y = 0
  LIST_COORD_LEFT_HIP.insert(0,(LEFT_HIP_X,LEFT_HIP_Y))
  LIST_COORD_LEFT_HIP.pop(29)

  CENTER_HIP_X = int((RIGHT_HIP_X+LEFT_HIP_X)/2)
  CENTER_HIP_Y = int((RIGHT_HIP_Y+LEFT_HIP_Y)/2) 
  LIST_COORD_CENTER_HIP.insert(0,(CENTER_HIP_X,CENTER_HIP_Y))
  LIST_COORD_CENTER_HIP.pop(29)

  RIGHT_KNEE = landmark_pose[26]
  RIGHT_KNEE_X = int(RIGHT_KNEE.x * image_width)
  RIGHT_KNEE_Y = int(RIGHT_KNEE.y * image_height)
  if (RIGHT_KNEE.visibility < 0.5):
    RIGHT_KNEE_X = 0
    RIGHT_KNEE_Y = 0
  LIST_COORD_RIGHT_KNEE.insert(0,(RIGHT_KNEE_X,RIGHT_KNEE_Y))
  LIST_COORD_RIGHT_KNEE.pop(29)
  
  LEFT_KNEE = landmark_pose[25]
  LEFT_KNEE_X = int(LEFT_KNEE.x * image_width)
  LEFT_KNEE_Y = int(LEFT_KNEE.y * image_height)
  if (LEFT_KNEE.visibility < 0.5):
    LEFT_KNEE_X = 0
    LEFT_KNEE_Y = 0
  LIST_COORD_LEFT_KNEE.insert(0,(LEFT_KNEE_X,LEFT_KNEE_Y))
  LIST_COORD_LEFT_KNEE.pop(29)

  RIGHT_ANKLE = landmark_pose[28]
  RIGHT_ANKLE_X = int(RIGHT_ANKLE.x * image_width)
  RIGHT_ANKLE_Y = int(RIGHT_ANKLE.y * image_height)
  if (RIGHT_ANKLE.visibility < 0.5):
    RIGHT_ANKLE_X = 0
    RIGHT_ANKLE_Y = 0
  LIST_COORD_RIGHT_ANKLE.insert(0,(RIGHT_ANKLE_X,RIGHT_ANKLE_Y))
  LIST_COORD_RIGHT_ANKLE.pop(29)
  
  LEFT_ANKLE = landmark_pose[27]
  LEFT_ANKLE_X = int(LEFT_ANKLE.x * image_width)
  LEFT_ANKLE_Y = int(LEFT_ANKLE.y * image_height)
  if (LEFT_ANKLE.visibility < 0.5):
    LEFT_ANKLE_X = 0
    LEFT_ANKLE_Y = 0 
  LIST_COORD_LEFT_ANKLE.insert(0,(LEFT_ANKLE_X,LEFT_ANKLE_Y))
  LIST_COORD_LEFT_ANKLE.pop(29)

  
  if (LIST_COORD_RIGHT_SHOULDER[0][0]!=0 and LIST_COORD_RIGHT_ELBOW[0][0]!=0):
    # 오른쪽 어깨 - 오른쪽 팔꿈치
    cv2.line(image, (LIST_COORD_RIGHT_SHOULDER[0][0] , LIST_COORD_RIGHT_SHOULDER[0][1]) , (LIST_COORD_RIGHT_ELBOW[0][0], LIST_COORD_RIGHT_ELBOW[0][1]), (0,0,255), 1, cv2.LINE_AA)
  
  if (LIST_COORD_RIGHT_WRIST[0][0]!=0 and LIST_COORD_RIGHT_ELBOW[0][0]!=0):
    # 오른쪽 손목 - 오른쪽 팔꿈치
    cv2.line(image, (LIST_COORD_RIGHT_WRIST[0][0] , LIST_COORD_RIGHT_WRIST[0][1]) , (LIST_COORD_RIGHT_ELBOW[0][0], LIST_COORD_RIGHT_ELBOW[0][1]), (0,0,255), 1, cv2.LINE_AA)
  
  if (LIST_COORD_RIGHT_SHOULDER[0][0]!=0 and LIST_COORD_CENTER_SHOULDER[0][0]!=0):
    # 오른쪽 어깨 - 중어깨
    cv2.line(image, (LIST_COORD_RIGHT_SHOULDER[0][0] , LIST_COORD_RIGHT_SHOULDER[0][1]) , (LIST_COORD_CENTER_SHOULDER[0][0], LIST_COORD_CENTER_SHOULDER[0][1]), (0,0,255), 1, cv2.LINE_AA)

  if (LIST_COORD_CENTER_SHOULDER[0][0]!=0 and LIST_COORD_LEFT_SHOULDER[0][0]!=0):
    # 중어깨 - 왼쪽 어깨
    cv2.line(image, (LIST_COORD_CENTER_SHOULDER[0][0] , LIST_COORD_CENTER_SHOULDER[0][1]) , (LIST_COORD_LEFT_SHOULDER[0][0], LIST_COORD_LEFT_SHOULDER[0][1]), (0,0,255), 1, cv2.LINE_AA)
  
  if (LIST_COORD_LEFT_SHOULDER[0][0]!=0 and LIST_COORD_LEFT_ELBOW[0][0]!=0):
    # 왼쪽 어깨 - 왼쪽 팔꿈치
    cv2.line(image, (LIST_COORD_LEFT_SHOULDER[0][0], LIST_COORD_LEFT_SHOULDER[0][1]) , (LIST_COORD_LEFT_ELBOW[0][0], LIST_COORD_LEFT_ELBOW[0][1]), (0,0,255), 1, cv2.LINE_AA)
  
  if (LIST_COORD_LEFT_ELBOW[0][0]!=0 and LIST_COORD_LEFT_WRIST[0][0]!=0):
    # 왼쪽 팔꿈치 - 왼쪽 손목
    cv2.line(image, (LIST_COORD_LEFT_ELBOW[0][0], LIST_COORD_LEFT_ELBOW[0][1]) , (LIST_COORD_LEFT_WRIST[0][0], LIST_COORD_LEFT_WRIST[0][1]), (0,0,255), 1, cv2.LINE_AA)
  if (LIST_COORD_CENTER_SHOULDER[0][0]!=0 and LIST_COORD_CENTER_HIP[0][0]!=0):  
    # 중어꺠 - 중덩이
    cv2.line(image, (LIST_COORD_CENTER_SHOULDER[0][0] , LIST_COORD_CENTER_SHOULDER[0][1]) , (LIST_COORD_CENTER_HIP[0][0], LIST_COORD_CENTER_HIP[0][1]), (0,0,255), 1, cv2.LINE_AA)
  if (LIST_COORD_RIGHT_HIP[0][0]!=0 and LIST_COORD_CENTER_HIP[0][0]!=0):
    # 오른쪽 엉덩이 - 중덩이
    cv2.line(image, (LIST_COORD_RIGHT_HIP[0][0] , LIST_COORD_RIGHT_HIP[0][1]) , (LIST_COORD_CENTER_HIP[0][0], LIST_COORD_CENTER_HIP[0][1]), (0,0,255), 1, cv2.LINE_AA)
  if (LIST_COORD_LEFT_HIP[0][0]!=0 and LIST_COORD_CENTER_HIP[0][0]!=0):
    # 왼쪽 엉덩이 - 중덩이
    cv2.line(image, (LIST_COORD_LEFT_HIP[0][0] , LIST_COORD_LEFT_HIP[0][1]) , (LIST_COORD_CENTER_HIP[0][0], LIST_COORD_CENTER_HIP[0][1]), (0,0,255), 1, cv2.LINE_AA)
  if (LIST_COORD_RIGHT_HIP[0][0]!=0 and LIST_COORD_RIGHT_KNEE[0][0]!=0):
    # 오른쪽 엉덩이 - 오른쪽 무릎
    cv2.line(image, (LIST_COORD_RIGHT_HIP[0][0] , LIST_COORD_RIGHT_HIP[0][1]) , (LIST_COORD_RIGHT_KNEE[0][0], LIST_COORD_RIGHT_KNEE[0][1]), (0,0,255), 1, cv2.LINE_AA)
  if (LIST_COORD_RIGHT_KNEE[0][0]!=0 and LIST_COORD_RIGHT_ANKLE[0][0]!=0):
    # 오른쪽 무릎 - 오른쪽 발목
    cv2.line(image, (LIST_COORD_RIGHT_KNEE[0][0] , LIST_COORD_RIGHT_KNEE[0][1]) , (LIST_COORD_RIGHT_ANKLE[0][0], LIST_COORD_RIGHT_ANKLE[0][1]), (0,0,255), 1, cv2.LINE_AA)
  if (LIST_COORD_LEFT_HIP[0][0]!=0 and LIST_COORD_LEFT_KNEE[0][0]!=0):
    # 왼쪽 엉덩이 - 왼쪽 무릎
    cv2.line(image, (LIST_COORD_LEFT_HIP[0][0], LIST_COORD_LEFT_HIP[0][1]) , (LIST_COORD_LEFT_KNEE[0][0], LIST_COORD_LEFT_KNEE[0][1]), (0,0,255), 1, cv2.LINE_AA)
  if (LIST_COORD_LEFT_KNEE[0][0]!=0 and LIST_COORD_LEFT_ANKLE[0][0]!=0):
    # 왼쪽 무릎 - 왼쪽 발목
    cv2.line(image, (LIST_COORD_LEFT_KNEE[0][0] , LIST_COORD_LEFT_KNEE[0][1]) , (LIST_COORD_LEFT_ANKLE[0][0], LIST_COORD_LEFT_ANKLE[0][1]), (0,0,255), 1, cv2.LINE_AA)

  for idx in range(MHI_DURATION):
    cv2.circle(image, (LIST_COORD_RIGHT_SHOULDER[idx][0],LIST_COORD_RIGHT_SHOULDER[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_RIGHT_ANKLE[idx][0],LIST_COORD_RIGHT_ANKLE[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_RIGHT_ELBOW[idx][0],LIST_COORD_RIGHT_ELBOW[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_RIGHT_HIP[idx][0],LIST_COORD_RIGHT_HIP[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_RIGHT_KNEE[idx][0],LIST_COORD_RIGHT_KNEE[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_RIGHT_WRIST[idx][0],LIST_COORD_RIGHT_WRIST[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_SHOULDER[idx][0],LIST_COORD_LEFT_SHOULDER[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_ANKLE[idx][0],LIST_COORD_LEFT_ANKLE[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_ELBOW[idx][0],LIST_COORD_LEFT_ELBOW[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_KNEE[idx][0],LIST_COORD_LEFT_KNEE[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_WRIST[idx][0],LIST_COORD_LEFT_WRIST[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_HIP[idx][0],LIST_COORD_LEFT_HIP[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 

  cv2.circle(image, (0,0), 18, (0,0,0),cv2.FILLED, cv2.LINE_AA)

  return image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_classes = train_data_set.num_classes
custom_model = CustomConvNet(num_classes=num_classes).to(device)
custom_model.load_state_dict(torch.load('actionModelv1.pt'))

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

with torch.no_grad():
  cap = cv2.VideoCapture(0)

  with mp_pose.Pose(min_detection_confidence=0.8,min_tracking_confidence=0.5) as pose:
    while (cap.isOpened()):
      ret, frame = cap.read()
      if ret==False:
        break

      cv2.imshow('Original', cv2.flip(frame, 1))

      # Define a transform to convert the image to tensor
      transform = transforms.ToTensor()

      frame_process = frame.copy()
      # frame_process = cv2.resize(frame_process, (160,160)) # 400,500으로 변환
      frame_process = cv2.resize(frame_process, None, fx=0.25 , fy=0.25) #0.5배로 축소
      image_height, image_width, _ = frame_process.shape

      frame_process.flags.writeable = False
      frame_process = cv2.cvtColor(frame_process, cv2.COLOR_BGR2RGB)
      results = pose.process(frame_process)

      frame_process.flags.writeable = True
      if results.pose_landmarks:
        landmark_pose = results.pose_landmarks.landmark

        frame_process = cv2.cvtColor(frame_process, cv2.COLOR_RGB2BGR)
        frame_process = InsertCoordinate(frame_process,landmark_pose)

      # Convert the image to PyTorch tensor
        tensor_img = transform(frame_process)
        tensor_img = tensor_img.unsqueeze(0)

        outputs = custom_model(tensor_img)
        first , predicted = torch.max(outputs.data, 1)

        action = None
        if (predicted == 0):
          action = "stand"
        else:
          action = "squat"

        if (first > 0.5):
          print("action : {}, prob : {}".format(action,first))
          cv2.putText(frame, str(predicted), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 3)

        frame_process = cv2.resize(frame_process,None,fx=4,fy=4)
        cv2.imshow('skeletonMHI', cv2.flip(frame_process,1))
      if (cv2.waitKey(10) & 0xFF == ord('q')):
        break

    cap.release()
    cv2.destroyAllWindows()
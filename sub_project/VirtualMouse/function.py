import re
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

def ExtractFinger(image,results,RADIUS,COLOR):
  point = results.multi_hand_landmarks[0].landmark[8]
  height, width = image.shape[0], image.shape[1]
  point_x,point_y = int(point.x * width),int(point.y * height)

  COLOR = (0,255,0)

  cv2.circle(image, (point_x,point_y), RADIUS , COLOR , cv2.FILLED , cv2.LINE_AA) #속이 꽉 찬 

  return image,point_x,point_y

def IsBetween(target_x,target_y,start_x,start_y,end_x,end_y):
  Between_x = False
  Between_y = False

  if (start_x <= target_x<= end_x):
    Between_x = True
  else:
    Between_x = False

  if (start_y <= target_y<= end_y):
    Between_y = True
  else:
    Between_y = False
  
  if (Between_x and Between_y):
    return True
  else:
    return False

def FindCoordList(image,opt_list,block,height_ratio,width_ratio):
  x, y = 0, 0
  # start_x, start_y, end_x, end_y
  for i in range(block):
    opt_list.append(x)
    opt_list.append(y)
    x += width_ratio
    opt_list.append(x)
    opt_list.append(height_ratio)
  return opt_list

def DrawRectangle(image,COLOR,THICKNESS,opt_list,block):
  for i in range(block):
    index = 4*i
    cv2.rectangle(image, (opt_list[index],opt_list[index+1]), (opt_list[index+2],opt_list[index+3]), COLOR, THICKNESS)

def DetectInRange(image,COLOR,point_x,point_y,opt_list,block):
  for i in range(block):
    index = 4*i 
    if (IsBetween(point_x,point_y,opt_list[index],opt_list[index+1],opt_list[index+2],opt_list[index+3])):
      cv2.rectangle(image, (opt_list[index],opt_list[index+1]), (opt_list[index+2],opt_list[index+3]), COLOR, cv2.FILLED)
  return image

def PutText(image,opt_list,function_list,GAP,COLOR):
  for i in range(len(function_list)):
    index = 4*i
    cv2.putText(image, function_list[i] , (opt_list[index]+GAP,opt_list[index+1]+GAP), cv2.FONT_HERSHEY_SIMPLEX, 1,COLOR, 2, cv2.LINE_AA)

def ChooseOption(image,point_x,point_y,block):
  height, width = image.shape[0], image.shape[1]

  height_ratio,width_ratio= int(height/7),int(width/block)

  COLOR = (216,168,74)
  TEXT_COLOR = (255,255,255)
  GAP = 50
  THICKNESS = 3

  opt_list = []
  FindCoordList(image,opt_list,block,height_ratio,width_ratio)
  DrawRectangle(image,COLOR,THICKNESS,opt_list,block)
  DetectInRange(image,COLOR,point_x,point_y,opt_list,block)

  function_list = ['Music','Record','Return','Feedback','Challenge']
  PutText(image,opt_list,function_list,GAP,TEXT_COLOR)

  # Music | Record | Return | Feedback | challenge
  
  
  #속이 채워진 사각형
  # cv2.rectangle(image, (300,100), (400,300), COLOR, cv2.FILLED)
with mp_hands.Hands(max_num_hands=1,model_complexity=0,min_detection_confidence=0.8,min_tracking_confidence=0.5) as hands:    
  while cap.isOpened():
    ret, frame = cap.read()
    
    if (ret == False):
      break

    # Recolor Feed
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False        
    image = cv2.flip(image,1)
    # Make Detections
    results = hands.process(image)

    image.flags.writeable = True   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Export coordinates
    # try:
    try:
      if results.multi_hand_landmarks:
        image, point_x, point_y = ExtractFinger(image,results,15,(0,255,0))
        ChooseOption(image,point_x,point_y,5)

    except:
      pass
    
    cv2.imshow('Virtual Mouse', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
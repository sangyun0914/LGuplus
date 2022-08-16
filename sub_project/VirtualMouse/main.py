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

def ChooseOption(image,point_x,point_y):
  height, width = image.shape[0], image.shape[1]
  height_ratio,width_ratio= int(height/7),int(width/5)

  COLOR = (216,168,74)
  THICKNESS = 3

  opt1_start_x,opt1_start_y,opt1_end_x,opt1_end_y = 0,0,width_ratio,height_ratio

  opt2_start_x,opt2_start_y,opt2_end_x,opt2_end_y = opt1_start_x+width_ratio,0,opt1_end_x+width_ratio,height_ratio

  opt3_start_x,opt3_start_y,opt3_end_x,opt3_end_y = opt2_start_x+width_ratio,0,opt2_end_x+width_ratio,height_ratio

  opt4_start_x,opt4_start_y,opt4_end_x,opt4_end_y = opt3_start_x+width_ratio,0,opt3_end_x+width_ratio,height_ratio
  
  opt5_start_x,opt5_start_y,opt5_end_x,opt5_end_y = opt4_start_x+width_ratio,0,width,height_ratio

  cv2.rectangle(image, (opt1_start_x,opt1_start_y), (opt1_end_x,opt1_end_y), COLOR, THICKNESS)
  cv2.rectangle(image, (opt2_start_x,opt2_start_y), (opt2_end_x,opt2_end_y), COLOR, THICKNESS)
  cv2.rectangle(image, (opt3_start_x,opt3_start_y), (opt3_end_x,opt3_end_y), COLOR, THICKNESS)
  cv2.rectangle(image, (opt4_start_x,opt4_start_y), (opt4_end_x,opt4_end_y), COLOR, THICKNESS)
  cv2.rectangle(image, (opt5_start_x,opt5_start_y), (opt5_end_x,opt5_end_y), COLOR, THICKNESS)

  if (IsBetween(point_x,point_y,opt1_start_x,opt1_start_y,opt1_end_x,opt1_end_y)):
    cv2.rectangle(image, (opt1_start_x,opt1_start_y), (opt1_end_x,opt1_end_y), COLOR, cv2.FILLED)

  elif (IsBetween(point_x,point_y,opt2_start_x,opt2_start_y,opt2_end_x,opt2_end_y)):
    cv2.rectangle(image, (opt2_start_x,opt2_start_y), (opt2_end_x,opt2_end_y), COLOR, cv2.FILLED)

  elif (IsBetween(point_x,point_y,opt3_start_x,opt3_start_y,opt3_end_x,opt3_end_y)):
    cv2.rectangle(image, (opt3_start_x,opt3_start_y), (opt3_end_x,opt3_end_y), COLOR, cv2.FILLED)
    
  elif (IsBetween(point_x,point_y,opt4_start_x,opt4_start_y,opt4_end_x,opt4_end_y)):
    cv2.rectangle(image, (opt4_start_x,opt4_start_y), (opt4_end_x,opt4_end_y), COLOR, cv2.FILLED)

  elif (IsBetween(point_x,point_y,opt5_start_x,opt5_start_y,opt5_end_x,opt5_end_y)):
    cv2.rectangle(image, (opt5_start_x,opt5_start_y), (opt5_end_x,opt5_end_y), COLOR, cv2.FILLED)
  
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
  
    # Make Detections
    results = hands.process(image)

    image.flags.writeable = True   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Export coordinates
    # try:
    try:
      if results.multi_hand_landmarks:
        image, point_x, point_y = ExtractFinger(image,results,5,(0,255,0))
        ChooseOption(image,point_x,point_y)

    except:
      pass
    
    image = cv2.flip(image,1)
    cv2.imshow('Virtual Mouse', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
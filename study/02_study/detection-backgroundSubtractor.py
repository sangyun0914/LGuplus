from charset_normalizer import detect
import numpy as np
import cv2

#-------------------------------------------------------------------------------------------------------------------------------
# [함수 정의]
def mouse_handler(event, x, y, flags, param):
  global drawing
  dst_img = src_img.copy()

  if event == cv2.EVENT_LBUTTONDOWN:
    drawing = True
    point_list.append((x,y))

  if drawing:
    previous_point = None
    for point in point_list:
      cv2.circle(dst_img,point,3,COLOR,cv2.FILLED)
      if previous_point:
        cv2.line(dst_img,previous_point,point,COLOR,THICKNESS,cv2.LINE_AA)
      previous_point = point
    
    next_point = (x,y)
    if len(point_list) == 4:
      next_point = point_list[0] # 첫번째 클릭한 지점
    
    cv2.line(dst_img,previous_point,next_point,COLOR,THICKNESS,cv2.LINE_AA)

  if len(point_list) == 4:
    cv2.putText(src_img, "press 'q' to start tracking", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

  cv2.imshow('select-region',dst_img)
#-------------------------------------------------------------------------------------------------------------------------------
# [좌표 지정 phase]
# video의 첫번째 프레임을 받아옴
cap = cv2.VideoCapture('video/sample.mp4')
ret_init,frame_init = cap.read()
cap.release()

# 색깔 두께 drawing
COLOR = (255,0,255) #색깔
THICKNESS = 3 #두께
drawing = False # 선을 그릴지 여부
point_list = [] # 사용자가 찍은 좌표 저장

src_img = frame_init


cv2.namedWindow('select-region')
cv2.setMouseCallback('select-region',mouse_handler)
cv2.imshow('select-region', src_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

#-------------------------------------------------------------------------------------------------------------------------------
# [관심 영역 좌표 지정]
LeftUp_x, LeftUp_y = point_list[0] #유저의 첫번째 좌표
RightUp_x, RightUp_y = point_list[1] #유저의 두번쨰 좌표
RightDown_x, RightDown_y = point_list[2] #유저의 세번째 좌표
LeftDown_x, LeftDown_y = point_list[3] #유저의 네번째 좌표

frame_x,frame_xw,frame_y,frame_yh = 0,0,0,0

frame_y = LeftUp_y if (LeftUp_y<RightUp_y) else RightUp_y
frame_yh = LeftDown_y if (LeftDown_y>RightDown_y) else RightDown_y
frame_x = LeftUp_x if (LeftUp_x<LeftDown_x) else LeftDown_x
frame_xw = RightUp_x if (RightUp_x>RightDown_x) else RightDown_x

#-------------------------------------------------------------------------------------------------------------------------------
# [이상감지 phase]
#video cap and make background subtractor
cap = cv2.VideoCapture('video/sample.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

# 최소 픽셀화소, 최소 높이, 최소 너비 정의
MINIMUM_AREA = 250
MINIMUM_HEIGHT = 3
MINIMUM_WIDTH = 3

while True:
  ret, frame = cap.read()

  if (ret == False):
    break;
  
  # 원본 프레임의 높이와 너비 정보를 받아옴
  height = frame.shape[0]
  width = frame.shape[1]

  # 원본 frame에서 사용자가 지정했던 좌표를 기준으로 관심영역 지정
  roi = frame[frame_y:frame_yh, frame_x:frame_xw]

  # 관심영역 gray로 변환
  frame_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

  fgmask = fgbg.apply(frame_gray)

  entrude = False

  # fgmask의 연결된 흰색부분 검출
  count, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)

  # 발견된 흰색영역 iterate
  for i in range(1, count):
    x, y, w, h, size = stats[i]
    
    # 만약 size와 높이와 너비가 정의했던 최소값보다도 작으면 영역을 그리지 않고 skip
    if size < MINIMUM_AREA or h<MINIMUM_HEIGHT or w<MINIMUM_WIDTH:
      continue
    
    entrude = True

    #검출된 부분을 원본영상(frame)에 빨간색으로 라벨링
    cv2.rectangle(frame, (frame_x+x, frame_y+y, w, h), (0, 0, 255), 2)

  if (entrude == True):
    cv2.putText(frame, "anomaly activitiy FOUND!", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
  else:
    cv2.putText(frame, "anomaly activitiy not found", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
  #사용자가 찍었던 좌표를 초록색 사각형으로 계속 보여줌
  cv2.rectangle(frame,(frame_x , frame_y , frame_xw-frame_x , frame_yh-frame_y),(0,255,0),10)

  # 원본 프레임에 detection을 적용한 부분 출력
  cv2.imshow('original',frame)

  # 전경 출력
  cv2.imshow('fgmask',fgmask)

  # 전경과 원본이 겹치지 않게 frame의 width만큼 움직임
  cv2.moveWindow('fgmask',width,0)

  if cv2.waitKey(1) == ord('q'):
    break;

cap.release()
cv2.destroyAllWindows()
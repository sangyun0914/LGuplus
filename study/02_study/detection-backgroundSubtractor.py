from charset_normalizer import detect
import numpy as np
import cv2

# fields
point_list = []
cap = cv2.VideoCapture('video/sample.mp4')
ret_init,frame_init = cap.read()
cap.release()

src_img = frame_init

COLOR = (255,0,255) #색깔
THICKNESS = 3 #두께
drawing = False # 선을 그릴지 여부

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
      pass
      next_point = point_list[0] # 첫번째 클릭한 지점
    
    cv2.line(dst_img,previous_point,next_point,COLOR,THICKNESS,cv2.LINE_AA)

  if len(point_list) == 4:
    pass

  cv2.imshow('img',dst_img)

# [function : show_result()]
def show_result():
  width, height = 530, 710
  src = np.float32(point_list)
  dst = np.array([[0,0] , [width,0] , [width,height] , [0,height]], dtype=np.float32)

  matrix = cv2.getPerspectiveTransform(src,dst) #Matrix를 얻어옴
  result = cv2.warpPerspective(src_img, matrix, (width,height)) #matrix 대로 변환을 함

  cv2.imshow('result',result)


cv2.namedWindow('img')
cv2.setMouseCallback('img',mouse_handler)
cv2.imshow('img', src_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

LeftUp_x, LeftUp_y = point_list[0]
RightUp_x, RightUp_y = point_list[1]
RightDown_x, RightDown_y = point_list[2]
LeftDown_x, LeftDown_y = point_list[3]

frame_x,frame_xw,frame_y,frame_yh = 0,0,0,0

frame_y = LeftUp_y if (LeftUp_y<RightUp_y) else RightUp_y
frame_yh = LeftDown_y if (LeftDown_y>RightDown_y) else RightDown_y
frame_x = LeftUp_x if (LeftUp_x<LeftDown_x) else LeftDown_x
frame_xw = RightUp_x if (RightUp_x>RightDown_x) else RightDown_x

print(point_list)
print("y : {} | yh : {} | x : {} | xw : {}".format(frame_y,frame_yh,frame_x,frame_xw))

cap = cv2.VideoCapture('video/sample.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

MINIMUM_AREA = 250
MINIMUM_HEIGHT = 3
MINIMUM_WIDTH = 3

while True:
  ret, frame = cap.read()

  if (ret == False):
    break;

  roi = frame[frame_y:frame_yh, frame_x:frame_xw]

  frame_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
  fgmask = fgbg.apply(frame_gray)

  # fgmask의 흰색부분 검출
  cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)

  for i in range(1, cnt):
    x, y, w, h, s = stats[i]
    
    if s < MINIMUM_AREA or h<MINIMUM_HEIGHT or w<MINIMUM_WIDTH:
      continue
    
    # print("height : {} | width : {}".format(h,w))

    cv2.rectangle(frame, (frame_x+x, frame_y+y, w, h), (0, 0, 255), 2)

  cv2.rectangle(frame,(frame_x , frame_y , frame_xw-frame_x , frame_yh-frame_y),(0,255,0),10)
  cv2.imshow('original',frame)
  cv2.imshow('fgmask',fgmask)
  # cv2.imshow('background',background)a

  if cv2.waitKey(1) == ord('q'):
    break;

cap.release()
cv2.destroyAllWindows()
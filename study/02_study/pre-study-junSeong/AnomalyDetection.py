from charset_normalizer import detect
import torch
import numpy as np
import cv2

#-------------------------------------------------------------------------------------------------------------------------------
# [함수 정의]
# 네 개의 좌표를 받아옴

class detection:
  def __init__(self):
    self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # or yolov5n - yolov5x6, custom
    self.classes = self.model.names
    self.device = 'cpu'

  def score_frame(self, frame):
    """
    Takes a single frame as input, and scores the frame using yolo5 model.
    :param frame: input frame in numpy/list/tuple format.
    :return: Labels and Coordinates of objects detected by model in the frame.
    """
    self.model.to(self.device)
    frame = [frame]
    results = self.model(frame)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord

  def plot_boxes(self, labels, cord, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """
    n = len(labels)
    count = 0
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        if labels[i] == 0:
            count += 1
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                # cv2.putText(frame, self.classes[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    return frame, count
  def __call__(self):
    pass
  
  def ProcessDetection(self,frame,roi):
    labels, cord = self.score_frame(roi)

    _, count = self.plot_boxes(labels, cord, roi)

    if (count != 0):
      return True

    else:
      return False

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
    cv2.putText(src_img, "press any button to start tracking", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

  cv2.imshow('select-region',dst_img)

# 선택받은 네 개의 좌표를 기준으로 사각형 연산
def defineRectangle(point_list):
  LeftUp_x, LeftUp_y = point_list[0] #유저의 첫번째 좌표
  RightUp_x, RightUp_y = point_list[1] #유저의 두번쨰 좌표
  RightDown_x, RightDown_y = point_list[2] #유저의 세번째 좌표
  LeftDown_x, LeftDown_y = point_list[3] #유저의 네번째 좌표

  frame_x,frame_xw,frame_y,frame_yh = 0,0,0,0

  frame_y = LeftUp_y if (LeftUp_y<RightUp_y) else RightUp_y
  frame_yh = LeftDown_y if (LeftDown_y>RightDown_y) else RightDown_y
  frame_x = LeftUp_x if (LeftUp_x<LeftDown_x) else LeftDown_x
  frame_xw = RightUp_x if (RightUp_x>RightDown_x) else RightDown_x

  return frame_x, frame_xw, frame_y, frame_yh


#-------------------------------------------------------------------------------------------------------------------------------
# [좌표 지정 phase]
# video의 첫번째 프레임을 받아옴
cap = cv2.VideoCapture('video/carandhuman.mp4')
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
frame_x,frame_xw,frame_y,frame_yh = defineRectangle(point_list)

#-------------------------------------------------------------------------------------------------------------------------------
# [이상감지 phase]
#video cap and make background subtractor
cap = cv2.VideoCapture('video/carandhuman.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

# 최소 픽셀화소, 최소 높이, 최소 너비 정의
MINIMUM_AREA = 250
MINIMUM_HEIGHT = 3
MINIMUM_WIDTH = 3

detector = detection()
detector()

while True:
  ret, frame = cap.read()

  if (ret == False):
    break;
  
  # 원본 프레임의 높이와 너비 정보를 받아옴
  height = frame.shape[0]
  width = frame.shape[1]

  # 원본 frame에서 사용자가 지정했던 좌표를 기준으로 관심영역 지정
  RegionOfInterest = frame[frame_y:frame_yh, frame_x:frame_xw]

  # 관심영역 gray로 변환
  frame_gray = cv2.cvtColor(RegionOfInterest, cv2.COLOR_BGR2GRAY)

  fgmask = fgbg.apply(frame_gray)

  entrude = False
  humanEncoming = False

  # fgmask의 연결된 흰색부분 검출
  count, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)
  #개수    라벨   검출된영역정보  무게중심 위치

  # 발견된 흰색영역 iterate
  for i in range(1, count):
    x, y, w, h, size = stats[i]
    
    # 만약 size와 높이와 너비가 정의했던 최소값보다도 작으면 영역을 그리지 않고 skip
    if size < MINIMUM_AREA or h<MINIMUM_HEIGHT or w<MINIMUM_WIDTH:
      continue
    
    # 위 조건으로 모두 걸렀는데도 진행된다면 침입이 감지된 것
    entrude = True
    humanEncoming = detector.ProcessDetection(frame,RegionOfInterest)
    #검출된 부분을 원본영상(frame)에 빨간색으로 라벨링
    cv2.rectangle(frame, (frame_x+x, frame_y+y, w, h), (0, 0, 255), 2)

  if (entrude == True and humanEncoming == True):
    cv2.putText(frame, "Human activitiy FOUND!", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    
  elif (entrude == True and humanEncoming == False):
    cv2.putText(frame, "Unknown activitiy FOUND!", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

  elif (entrude == False and humanEncoming == False):
    cv2.putText(frame, "Activity not FOUND..", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

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
import multiprocessing as mp
from multiprocessing import Lock
from statistics import median
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np
import time
import copy
from matplotlib import pyplot as plt

# ---------------------------------------------------------------------------------------------------------
# [세로 hcout x 가로 wcount view 만들기]
def create_image_multiple(h, w, d, hcout, wcount):
  image = np.zeros((h*hcout, w*wcount,  d), np.uint8)
  color = tuple(reversed((0,0,0)))
  image[:] = color

  return image

# [각 세션의 어느 부분에 뭘 넣을지 결정]
def showMultiImage(dst, src, h, w, d, col, row):
  # 3 color (color)
  if  d==3:
    dst[(col*h):(col*h)+h, (row*w):(row*w)+w] = src[0:h, 0:w]
  # 1 color (grey)
  elif d==1:
    dst[(col*h):(col*h)+h, (row*w):(row*w)+w, 0] = src[0:h, 0:w]
    dst[(col*h):(col*h)+h, (row*w):(row*w)+w, 1] = src[0:h, 0:w]
    dst[(col*h):(col*h)+h, (row*w):(row*w)+w, 2] = src[0:h, 0:w]

# ---------------------------------------------------------------------------------------------------------
# [frame처리 함수 : canny]
'''
[Canny 함수의 세부과정]
1. Noise reduction using Gaussian filter 
- 노이즈 제거 이유 : 노이즈에서 화소의 갑작스런 강도 변화를 edge라고 잘못 판달할 수 있기 때문

2. Gradient calculation along the horizontal and vertical axis 
- Sobel mask의 수학적 정의에 따라 x, y방향의 gradient를 종합한 Edge Gradient이미지를 생성 -> edge 추출

3. Non-Maximum Suppression (NMS) of false edges 
- 화소 강도 차이가 큰 edge를 제외하고는 모두 억제하는 것
- NMS는 욜로의 bounding box의 suppression 용도로도 사용됨 (yolo)

4. Double thresholding for segregating strong and weak edges 
- double thresholding이라는 이름답게 2개의 임계값을 가짐
- (ex) th1 = 40 | th2 = 150 이라고 했을 때
- 화소가 150이상인 것은 강한강도의 edge이기 때문에 살림
- 화소가 40과 150 사이의 값이면 th2이상의 화소와 연결됐을 경우 살리지만, 혼자 동떨어진 값이면 버림
- 화소가 40이하인 것은 약한강도의 edge로 판단하고 버림

5. Edge tracking by hysteresis
- (4)에서 thresholding 했던 대로 edge 트랙킹
'''
def frame_process_canny(frame):
  frame_processed = cv2.Canny(frame,100,200)

  return frame_processed

# [frame처리 함수 : blur]
def frame_process_faceblur(frame,faceCascade):
  # frame gray scale로 변환
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # face detect
  faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
  )

  # detect된 face에 모자이크 처리
  for (x, y, w, h) in faces: # x : start position of x | y : start position of y | w : width | h : height
    rate=10
    mosaic = frame[y:y+h, x:x+w] # 감지된 얼굴 추출

    mosaic = cv2.resize(mosaic, (w//rate, h//rate)) # 1/rate 비율로 축소 -> 모자이크가 더 효과적으로 적용되기 위해
    mosaic = cv2.resize(mosaic, (w,h), interpolation=cv2.INTER_AREA)
    
    mosaic = cv2.GaussianBlur(mosaic , (7,7) , 0) #resize된 mosiac에 blur 처리를 줘서 좀 더 모자이크 효과를 줌

    frame[y:y+h, x:x+w] = mosaic # 실제 프레임에 모자이크 적용

  return frame

# [frame처리 함수 : background 전경]
def frame_process_background_front(frame):
  # hh, ww = frame.shape[:2]

  # [threshold 적용]
  # 하한값과 상한값 정하기
  lower = np.array([180,180,180])
  upper = np.array([255, 255, 255])

  # frame에 상한값 하한 값 적용
  thresh = cv2.inRange(frame, lower, upper)

  # [morphology 적용]
  # 타원형 형태로 20x20 kernel array 생성
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))

  # thresh에 kernel 적용
  morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # thresh의 노이즈 제거

  # [morphology 적용된 이미지 뒤집기]
  mask = 255 - morph # 흑백 전환

  # [frame에 mask 적용]
  result = cv2.bitwise_and(frame, frame, mask=mask) # frame & frame

  return result

def frame_process_background_rear(frame, fgbg):
  fgmask = fgbg.apply(frame)
  back = fgbg.getBackgroundImage()

  return back

# [frame처리 함수 : blur]
'''
[이미지 블러처리] : 이미지에 커널(마스크)를 컨볼루션하여 블러닝효과를 줌 (활용하면 샤프닝효과도 가능)
- 1. Averaging blurring : 커널의 가중치가 모두 동일
- cv2.blur(src, (x,x))
- src : blur 처리할 frame
- (x,x) : 커널 사이즈

- 2. Gaussian Blurring : 커널의 가중치가 중심으로 갈수록 커짐
- cv2.GaussianBlur(src, (x,x), sigma)
- src : blur 처리할 frame
- (x,x) : 커널 사이즈
- sigma : 시그마 값을 얼마나 넣을 것인지

- 3. Median Blurring : 지정한 커널 크기 내의 픽셀을 크기순으로 정렬한 후 중간값을 뽑아서 픽셀값으로 사용
- cv2.medianBlur(src, x)
- src : blur 처리할 frame
- x : 커널 사이즈 (x,x)

'''
def frame_process_blur(frame):
  grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  grayframe = cv2.equalizeHist(grayframe)

  # median blur 적용
  blur = cv2.medianBlur(grayframe,5)

  return blur

# [frame처리 함수 : threshold]
'''
[이미지 임계처리] : 이미지 이진화의 방법인 simple thresholding 이용
- cv2.threshold(src, thresh, maxval, type) → retval, dst
- src : input image로 single channel(gray-scale) image를 넣어줌
- thresh : 임계값
- maxval : 임계값을 넘었을 때 적용할 값
- type : thresholding type
-      cv2.THRESH_BINARY
-      cv2.THRESH_BINARY_INV
-      cv2.THRESH_TRUNC
-      cv2.THRESH_TOZERO
-      cv2.THRESH_TOZERO_INV
- 문제점 : 임계값을 이미지 전체에 적용하여 처리하기 때문에 하나의 이미지에 음영이 다르면 일부 영역이 모두 흰색 또는 검정색으로 보여지게 됨

- cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
- src : grayscale image
- maxValue : 임계값
- adaptiveMethod : thresholding value를 결정하는 계산 방법
- thresholdType : threshold type
- blockSize : thresholding을 적용할 영역 사이즈
- C : 평균이나 가중평균에서 차감할 값
- blockSize나 C는 어떻게 결정?
'''
def frame_process_threshold(frame):
  # gray로 바꾸기
  grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # gray로 바뀐 이미지 평준화
  grayframe = cv2.equalizeHist(grayframe)

  # threshold를 적용하기 전 medianblur 적용 (무작위 노이즈를 제거하는데 효과적)
  median_blur = cv2.medianBlur(grayframe, 9)

  # threshold 적용
  thresholdFrame = cv2.adaptiveThreshold(median_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,2)

  return thresholdFrame

# ---------------------------------------------------------------------------------------------------------
# [thread executing function : yolo]
def yolo():
  capture = cv2.VideoCapture(0)

  while True:
    status, frame = capture.read()

    bbox, label, conf = cv.detect_common_objects(frame,confidence=0.5, nms_thresh=0.3, model='yolov3', enable_gpu=False)
    out = draw_bbox(frame, bbox, label, conf, write_conf=True)

    frame = cv2.resize(frame, None, fx=0.25, fy=0.25) #0.5배

    cv2.imshow('yolo-object detection', frame)
    cv2.moveWindow('yolo-object detection', 500 , 0)

    if cv2.waitKey(1) == ord('q'):
      break

  capture.release()
  cv2.destroyAllWindows()

# [thread executing function : main] ------------------------------------------------------------------------------------------
def main():
  # [카메라 가져오기]
  cap = cv2.VideoCapture(0)

  # [face detection을 위한 모델 가져오기]
  faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

  # 글자를 넣기 위한 변수 설정
  SCALE = 1
  COLOR_WHITE = (255,255,255)
  COLOR_RED = (0,0,255)
  COLOR_BLACK = (0,0,0) 
  THICKNESS = 2

  #
  fgbg = cv2.createBackgroundSubtractorMOG2()

  # 카메라 안 열리면 exit
  if not cap.isOpened():
    exit()

  # [열린 카메라로 프레임 처리 시작]
  while cap.isOpened():

    #프레임 캡쳐
    ret, frame_original = cap.read()
    width, height , channel = frame_original.shape # original frame의 width, height, channel 정보를 받아옴

    frame_original = cv2.resize(frame_original, (0, 0), None, .25, .25) # 본래 프레임에 0.25배 적용 -> 화면 분할로 들어갈 것이기 때문

    frame_faceblur = frame_original.copy() #blur frame 처리
    frame_canny = frame_original.copy() #canny frame 처리
    frame_background_front= frame_original.copy() #background_front frame 처리
    frame_background_rear = frame_original.copy() #background_rear frame 처리
    frame_blur = frame_original.copy() #blur 처리
    frame_threshold = frame_original.copy() # threshold 처리
    empty = np.zeros((width,height,channel), np.uint8) # original frame에서 받아온 shape 정보로 empty view 만들기 (검은화면)

    # process frame
    frame_canny = frame_process_canny(frame_canny)
    frame_faceblur = frame_process_faceblur(frame_faceblur,faceCascade)
    frame_blur = frame_process_blur(frame_blur)
    frame_threshold = frame_process_threshold(frame_threshold)
    frame_background_front = frame_process_background_front(frame_background_front)
    fgmask = fgbg.apply(frame_background_rear)
    frame_background_rear = fgbg.getBackgroundImage()

    # 이미지 높이
    height = frame_original.shape[0]
    # 이미지 넓이
    width = frame_original.shape[1]
    # 이미지 색상 크기
    depth = frame_original.shape[2]

    # 화면에 표시할 이미지 만들기 ( 4 x 2 )
    view = create_image_multiple(height, width, depth, 4, 2)

    # 원하는 위치에 화면 넣기
    showMultiImage(view, frame_background_front, height,width,depth,0,0)
    showMultiImage(view, frame_background_rear, height, width , depth , 0, 1)
    showMultiImage(view, frame_faceblur, height, width, depth, 1, 0)
    showMultiImage(view, frame_canny, height, width, 1, 1, 1)
    showMultiImage(view, frame_original, height, width, depth, 2, 0)
    showMultiImage(view, empty, height, width, depth, 2, 1)
    showMultiImage(view, frame_threshold, height, width, 1, 3, 0)
    showMultiImage(view, frame_blur, height, width, 1, 3, 1)

    # 글자 넣기
    cv2.putText(view, "background-front", (20,50), cv2.FONT_HERSHEY_SIMPLEX, SCALE, COLOR_BLACK, THICKNESS)
    cv2.putText(view, "background-back", (500,50), cv2.FONT_HERSHEY_SIMPLEX, SCALE, COLOR_BLACK, THICKNESS)
    cv2.putText(view, "faceblur", (20,300), cv2.FONT_HERSHEY_SIMPLEX, SCALE, COLOR_BLACK, THICKNESS)
    cv2.putText(view, "canny", (500,300), cv2.FONT_HERSHEY_SIMPLEX, SCALE, COLOR_RED, THICKNESS)
    cv2.putText(view, "original", (20,575), cv2.FONT_HERSHEY_SIMPLEX, SCALE, COLOR_BLACK, THICKNESS)
    cv2.putText(view, "threshold", (20,850), cv2.FONT_HERSHEY_SIMPLEX, SCALE, COLOR_RED, THICKNESS)
    cv2.putText(view, "blur", (500,850), cv2.FONT_HERSHEY_SIMPLEX, SCALE, COLOR_RED, THICKNESS)

    # 화면 표시
    cv2.imshow('Realtime-image-processing',view)

    if cv2.waitKey(1) == ord('q'):
      break

  # [resource release]
  cap.release()
  cv2.destroyAllWindows()

#--------------------------------------------------------------------
# [start]
if __name__ == "__main__":
  threadMain = mp.Process(target=main, args=())
  threadYolo = mp.Process(target=yolo, args=())

  threadMain.start()
  threadYolo.start()
#--------------------------------------------------------------------

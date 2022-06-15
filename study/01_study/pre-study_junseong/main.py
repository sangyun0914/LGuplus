import multiprocessing as mp
from multiprocessing import Lock
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

# [frame처리 함수 : canny]
def frame_process_canny(frame):
  frame_processed = cv2.Canny(frame,100,200)

  return frame_processed

# [frame처리 함수 : blur]
def frame_process_faceblur(frame,faceCascade):
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
	)

  for (x, y, w, h) in faces:
    rate=10
    # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    target = frame[y:y+h, x:x+w]   # 관심영역 지정
    target = cv2.resize(target, (w//rate, h//rate)) # 1/rate 비율로 축소
    target = cv2.resize(target, (w,h), interpolation=cv2.INTER_AREA)  
    frame[y:y+h, x:x+w] = target   # 원본 이미지에 적용

  return frame

# [frame처리 함수 : yolo]
def frame_process_yolo(frame):
  bbox, label, conf = cv.detect_common_objects(frame,confidence=0.5, nms_thresh=0.3, model='yolov3-tiny', enable_gpu=False)
  out = draw_bbox(frame, bbox, label, conf, write_conf=True)

# [frame처리 함수 : background 전경]
def frame_process_background_front(frame):
  hh, ww = frame.shape[:2]

# threshold on white
# Define lower and uppper limits
  lower = np.array([200, 200, 200])
  upper = np.array([255, 255, 255])

# Create mask to only select black
  thresh = cv2.inRange(frame, lower, upper)

# apply morphology
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
  morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# invert morp image
  mask = 255 - morph

# apply mask to image
  result = cv2.bitwise_and(frame, frame, mask=mask)

  return result

def frame_process_background_rear(frame):
  pass

# [frame처리 함수 : blur]
def frame_process_blur(frame):
  grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  grayframe = cv2.equalizeHist(grayframe)

  # median blur
  blur = cv2.medianBlur(grayframe,5)

  return blur

# [frame처리 함수 : threshold]
def frame_process_threshold(frame):
  grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  grayframe = cv2.equalizeHist(grayframe)

  # median blur
  blur = cv2.medianBlur(grayframe,5)

  # apply threshold
  ret, th1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

  return th1
  
# [thread executing function : yolo]
def yolo():
  capture = cv2.VideoCapture(0)

  while True:
    if capture.isOpened():
      status, frame = capture.read()
      bbox, label, conf = cv.detect_common_objects(frame,confidence=0.5, nms_thresh=0.3, model='yolov3', enable_gpu=False)
      out = draw_bbox(frame, bbox, label, conf, write_conf=True)

      # 이미지 높이
      height = frame.shape[0]
      # 이미지 넓이
      width = frame.shape[1]
      # 이미지 색상 크기
      depth = frame.shape[2]
      
      if status:
        cv2.imshow('yolo', frame)

      else:
        break

      if cv2.waitKey(1) == ord('q'):
        break

  capture.release()
  cv2.destroyAllWindows()

# [thread executing function : main] ------------------------------------------------------------------------------------------
def main():
  # --------------------------------------------------------------------------------------------------------
  # [카메라 가져오기]
  cap = cv2.VideoCapture(0)
  faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

  if not cap.isOpened():
    exit()

  # --------------------------------------------------------------------------------------------------------
  # [열린 카메라로 프레임 처리 시작]
  while cap.isOpened():

    #프레임 캡쳐
    ret, frame_original = cap.read()
    width, height , channel = frame_original.shape # original frame의 width, height, channel 정보를 받아옴

    frame_original = cv2.resize(frame_original, (0, 0), None, .25, .25)
    frame_gray = cv2.cvtColor(frame_original, cv2.COLOR_RGB2GRAY)

    frame_faceblur = frame_original.copy() #blur frame 처리
    frame_yolo = frame_original.copy() #yolo frame 처리
    frame_canny = frame_original.copy() #canny frame 처리
    frame_background_front= frame_original.copy() #background frame 처리
    frame_background_rear = frame_original.copy()
    frame_blur = frame_original.copy()
    frame_threshold = frame_original.copy()
    empty = np.zeros((width,height,channel), np.uint8) # original frame에서 받아온 shape 정보로 empty view 만들기 (검은화면)

    # process frame
    frame_canny = frame_process_canny(frame_canny)
    frame_faceblur = frame_process_faceblur(frame_faceblur,faceCascade)
    frame_blur = frame_process_blur(frame_blur)
    frame_threshold = frame_process_threshold(frame_threshold)
    frame_background_front = frame_process_background_front(frame_background_front)

    # 이미지 높이
    height = frame_original.shape[0]
    # 이미지 넓이
    width = frame_original.shape[1]
    # 이미지 색상 크기
    depth = frame_original.shape[2]

    # 화면에 표시할 이미지 만들기 ( 2 x 2 )
    dstimage = create_image_multiple(height, width, depth, 4, 2)

    # 원하는 위치에 화면 넣기
    showMultiImage(dstimage, frame_background_front, height,width,depth,0,0)
    showMultiImage(dstimage, frame_background_rear, height,width,depth,0,1)
    showMultiImage(dstimage, frame_faceblur, height, width, depth, 1, 0)
    showMultiImage(dstimage, frame_canny, height, width, 1, 1, 1)
    showMultiImage(dstimage, frame_original, height, width, depth, 2, 0)
    showMultiImage(dstimage, empty, height, width, depth, 2, 1)
    showMultiImage(dstimage, frame_threshold, height, width, 1, 3, 0)
    showMultiImage(dstimage, frame_blur, height, width, 1, 3, 1)

    # 화면 표시
    cv2.imshow('Realtime-processing',dstimage)

    if cv2.waitKey(1) == ord('q'):
      break
  # --------------------------------------------------------------------------------------------------------
  # [resource release]
  cap.release()
  cv2.destroyAllWindows()

#---------------------------------------
# [start]
if __name__ == "__main__":
  threadMain = mp.Process(target=main, args=())
  threadYolo = mp.Process(target=yolo, args=())

  threadMain.start()
  threadYolo.start()
#---------------------------------------
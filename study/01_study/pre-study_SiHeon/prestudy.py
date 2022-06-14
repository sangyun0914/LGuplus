import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np
import time
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)       #webcam을 연결
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if not cap.isOpened():      #webcam 연결 실패시
    raise IOError("Cannot open webcam")
    exit()
prevTime = 0

while cap.isOpened():    #webcam 연결 시 실행
    curTime = time.time()

    sec = curTime - prevTime
    prevTime = curTime

    ret, frame = cap.read()     #프레임별 캡쳐
    frame1 = frame
    frame2 = frame
    frame3 = frame
    frame4 = frame

    edges=cv2.Canny(frame3,100,200)
    video = cv2.resize(frame, (0, 0), None, .5, .5)     #일반 웹캠
    video1 = cv2.resize(frame1, (0, 0), None, .5, .5)   #얼굴인식 및 블러
    video2 = cv2.resize(frame2, (0, 0), None, .5, .5)   #yolo
    video3 = cv2.resize(edges, (0, 0), None, .5, .5)    #edge
    video4 = cv2.resize(frame4, (0, 0), None, .5, .5)

    video3=cv2.cvtColor(video3,cv2.COLOR_GRAY2BGR)

    bbox, label, conf = cv.detect_common_objects(video2)
    out = draw_bbox(video2, bbox, label, conf, write_conf=True)

    gray = cv2.cvtColor(video1, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
	)

    for (x, y, w, h) in faces:
        rate=10
        cv2.rectangle(video1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi = video1[y:y+h, x:x+w]   # 관심영역 지정
        roi = cv2.resize(roi, (w//rate, h//rate)) # 1/rate 비율로 축소
        # 원래 크기로 확대
        roi = cv2.resize(roi, (w,h), interpolation=cv2.INTER_AREA)  
        video1[y:y+h, x:x+w] = roi   # 원본 이미지에 적용

    numpy_horizontal = np.hstack((video, video1))
    numpy_horizontal1 = np.hstack((video2, video3))
    numpy_vertical = np.vstack((numpy_horizontal, numpy_horizontal1))

    fps = 1/(sec)

    # 디버그 메시지로 확인해보기

    # 프레임 수를 문자열에 저장
    str = "FPS : %0.1f" % fps

    # 표시
    print(str)

    cv2.imshow('webcam', numpy_vertical)     
    #cv2.imshow('ss',edges)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):       #키보드로 q를 누를경우 종료 **ord함수는 문자의 유니코드값을 리턴
        break

cap.release()       #카메라 사용 종료
cv2.destroyAllWindows()     #창 닫기
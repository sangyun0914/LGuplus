import cv2
import numpy as np

# 웹캠 설정
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while(True):
    ret, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 흑백으로 변환

    # threshold    
    ret, th = cv2.threshold(gray ,127, 255, cv2.THRESH_BINARY) # 임계값(127) 넘으면 백, 아니면 흑으로 바꿈
    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2) # 이미지 영역별로 이진화 진행, 영역의 평균값을 임계값으로 설정, 영역(블록)사이즈 = 15

    # blur
    blur = cv2.blur(frame, (5, 5)) # 5 * 5 박스내 픽셀 평균값
    gblur = cv2.GaussianBlur(frame, (15,15), 0) # 2차원 가우시안 분포 이용
    mblur = cv2.medianBlur(frame, 7) # 박스 내 픽셀들의 중간값, 박스 사이즈 = 7 
    bfblur = cv2.bilateralFilter(frame, 10, 75, 75) # 윤곽선은 살리고 나머지는 블러 처리

    # canny edge detection
    canny = cv2.Canny(gray, 70, 200)

    # face detection
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(frame)

    # mosaic
    mosaic = frame
    for (x, y, w, h) in faces:
        roi = mosaic[y:y+h, x:x+w]   # 관심영역 지정
        roi = cv2.resize(roi, (w//15, h//15)) # 1/20 비율로 축소
        # 원래 크기로 확대
        roi = cv2.resize(roi, (w,h), interpolation=cv2.INTER_AREA)  
        mosaic[y:y+h, x:x+w] = roi # 원본 이미지에 적용

    result1 = np.hstack((th, th2, canny))
    result2 = np.hstack((blur, gblur, mosaic))
    cv2.imshow('1', result1)
    cv2.imshow('2', result2)

    # 키보드 입력시 종료
    if cv2.waitKey(33) > 0:
        break

# 웹캠 종료, 모든 창 종료
capture.release()
cv2.destroyAllWindows()
 
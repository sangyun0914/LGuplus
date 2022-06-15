import cv2
import numpy as np

#웹캠 설정
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#배경 추출 알고리즘 함수, 오랜 시간 동안 변하지 않는 픽셀들을 배경으로 판단하는 방식
mask = cv2.bgsegm.createBackgroundSubtractorMOG()

# 배경 영상 등록
ret, back1 = capture.read()
    
# 연산 속도를 높이기 위해 그레이스케일 영상으로 변환
back1 = cv2.cvtColor(back1, cv2.COLOR_BGR2GRAY)

while(True):
    ret, frame = capture.read()

    # 현재 프레임 영상 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # absdiff는 차 영상에 절대값
    diff = cv2.absdiff(gray, back1)

    # 차이가 30이상 1, 40보다 작으면 0 -> 필터 역할
    _, diff2 = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    diff2 = cv2.cvtColor(diff2, cv2.COLOR_GRAY2BGR)
    _, filter1 = cv2.threshold(diff, 40, 1, cv2.THRESH_BINARY)

    #필터를 3채널로 만듦
    filter2 = cv2.merge((filter1, filter1, filter1))

    #전경 추출, 흑백
    back2 = mask.apply(frame)
    back2 = cv2.cvtColor(back2, cv2.COLOR_GRAY2BGR)

    #원본 영상에서 전경 추출을 위해 필터 생성
    _, filter3 = cv2.threshold(back2, 30, 1, cv2.THRESH_BINARY)

    #영상 두개 한번에 보여주기 위해 합침
    numpy_horizontal = np.hstack((diff2, frame * filter2, back2, frame * filter3))

    #전경 추출, 흑백/컬러
    cv2.imshow('background', numpy_horizontal)

    #키보드 입력시 종료
    if cv2.waitKey(33) > 0:
        break

#웹캠 종료, 모든 창 종료
capture.release()
cv2.destroyAllWindows()

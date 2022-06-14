#https://deep-learning-study.tistory.com/272

import cv2

#웹캠 설정
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 배경 영상 등록
ret, back = capture.read()
    
# 연산 속도를 높이기 위해 그레이스케일 영상으로 변환
back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)

# 비디오 매 프레임 처리
while True:
    ret, frame = capture.read()
    
    # 현재 프레임 영상 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # absdiff는 차 영상에 절대값
    diff = cv2.absdiff(gray, back)

    # 차이가 30이상 1, 40보다 작으면 0 -> 필터 역할
    _, diff = cv2.threshold(diff, 40, 1, cv2.THRESH_BINARY)

    #필터를 3채널로 만듦
    diff3 = cv2.merge((diff, diff, diff))

    #원래 프레임에 필터를 곱해서 전경 부분만 추출
    cv2.imshow('back', diff3 * frame)

    if cv2.waitKey(33) > 0:
        break

capture.release()
cv2.destroyAllWindows()
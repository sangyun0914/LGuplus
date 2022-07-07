# https://en.wikipedia.org/wiki/Motion_History_Images

import cv2
import numpy as np

# 웹캠 설정
cap = cv2.VideoCapture(0)

# 타우, 임계값 설정
tau = 60
threshold = 15

# 프로그램 시작할때 이전 프레임이 없기 때문에 루프 돌기 전에 그레이 스케일로 변환한 프레임을 이전 프레임으로 저장
_, previous_frame = cap.read()
previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

# motion history image, 웹캠과 똑같은 해상도의 크기로 0으로 채워진 넘파이 배열로 초기화
mhi = np.zeros(
    shape=(previous_frame.shape[0], previous_frame.shape[1]), dtype=np.uint8)

while cap.isOpened():
    success, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 프레임 그레이 스케일로 변환
    diff = cv2.absdiff(gray, previous_frame)  # 이전 프레임과 차이 계산
    # 임계값 이상으로 차이가 나는 픽셀 부분을 1로 설정한 마스크1 생성
    _, mask1 = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
    mask2 = 1 - mask1  # 마스크1을 반전시킨 마스크2 생성
    np.place(mhi, mask1, tau)  # 마스크1의 1에 해당하는 픽셀값을 타우값으로 변경
    zero = mhi == 0  # 언더플로우 방지하기 위해 미리 픽셀값이 0인 부분에 1을 미리 더해주기 위한 제로 마스크 생성
    # 픽셀값이 0인 부분은 미리 1로 만들어둬서 나중에 마스크2를 빼더라도 언더플로우가 발생하지 않도록 함
    np.place(mhi, zero, 1)
    mhi = mhi - mask2  # 픽셀값이 변하지 않는 부분에 해당하는 마스크2의 1에 해당하는 픽셀들은 mhi에서 1 감소시켜줌

    cv2.imshow('frame', mhi)

    # 현재 프레임을 이전 프레임으로 설정
    previous_frame = gray

    if cv2.waitKey(10) > 0:
        break

cap.release()

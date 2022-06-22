# https://www.youtube.com/watch?v=hMIrQdX4BkE
# https://www.youtube.com/watch?v=ticZclUYy88
# https://jstar0525.tistory.com/2

import cv2
import numpy as np

video = cv2.VideoCapture('test2.mp4')
mask = cv2.bgsegm.createBackgroundSubtractorMOG()
# mask = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# Mouse 클릭된 상태 (false = 클릭 x / true = 클릭 o) : 마우스 눌렀을때 true로, 뗏을때 false로
click = False
rec = False
x1, y1 = -1, -1
x2, y2 = -1, -1
img = None


def set_roi(event, x, y, flags, param):
    global x1, y1, click, rec, x2, y2

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 상태
        click = True
        rec = False
        x1, y1 = x, y
        print((x1, y1))

    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 이동
        if click == True:  # 마우스를 누른 상태 일경우
            rec = True
            x2, y2 = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        click = False  # 마우스를 때면 상태 변경
        rec = True
        x2, y2 = x, y
        print((x2, y2))


def detection(x, y):
    if x < x1 or x > x2 or y < y1 or y > y2:
        return False
    else:
        return True


def isHuman(w, h, area):
    if h > 2 * w and area > 10:
        return True
    else:
        return False


cv2.namedWindow('frame')
cv2.setMouseCallback('frame', set_roi)

while(True):
    ret, frame = video.read()
    # print(frame.shape)
    # (576, 768, 3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur = cv2.dilate(gray, kernel=None)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    back = mask.apply(blur)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(back)

    flag = False
    for s in stats:
        if flag:
            if detection(s[0], s[1]) and isHuman(s[2], s[3], s[4]):
                # 인식되었을때 화면 전체 빨간 테두리 알림 표시
                cv2.rectangle(
                    frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)
            cv2.rectangle(frame, (s[0], s[1]),
                          (s[0] + s[2], s[1]+s[3]), (255, 0, 0), 1)
        else:
            flag = True

    if rec:  # roi 표시
        cv2.rectangle(frame, (x1, y1),
                      (x2, y2), (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) > 0:
        break

video.release()
cv2.destroyAllWindows()

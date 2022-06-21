# https://www.youtube.com/watch?v=hMIrQdX4BkE
# https://www.youtube.com/watch?v=ticZclUYy88
# https://jstar0525.tistory.com/2


import cv2
# import labeling as mylabel
import numpy as np
# import sys

# sys.setrecursionlimit(10**7)

video = cv2.VideoCapture('test2.mp4')
mask = cv2.bgsegm.createBackgroundSubtractorMOG()
# mask = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

while(True):
    ret, frame = video.read()
    # print(frame.shape)
    # (576, 768, 3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur = cv2.dilate(gray, kernel=None)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    back = mask.apply(blur)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(back)
    print(ret - 1)

    for s in stats:
        cv2.rectangle(frame, (s[0], s[1]),
                      (s[0] + s[2], s[1]+s[3]), (255, 0, 0), 3)

    # print(len(mylabel.labeling(back)))

    result = np.hstack((frame, cv2.cvtColor(back, cv2.COLOR_GRAY2BGR)))
    cv2.imshow('back', result)
    if cv2.waitKey(1) > 0:
        break

video.release()
cv2.destroyAllWindows()

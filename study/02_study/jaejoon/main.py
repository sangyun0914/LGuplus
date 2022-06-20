import cv2
import numpy as np

video = cv2.VideoCapture('test2.mp4')
mask = cv2.bgsegm.createBackgroundSubtractorMOG()
# mask = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

while(True):
    ret, frame = video.read()
    # print(frame.shape)
    # (576, 768, 3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    back = mask.apply(blur)
    result = np.hstack((frame, cv2.cvtColor(back, cv2.COLOR_GRAY2BGR)))
    cv2.imshow('back', result)
    if cv2.waitKey(1) > 0:
        break

video.release()
cv2.destroyAllWindows()

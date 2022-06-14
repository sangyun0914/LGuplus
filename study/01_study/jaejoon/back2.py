import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mask = cv2.createBackgroundSubtractorMOG2()

while(True):
    ret, frame = capture.read()
    back = mask.apply(frame)

    _, back = cv2.threshold(back, 30, 1, cv2.THRESH_BINARY)
    back = cv2.merge((back, back, back))
    cv2.imshow('back2', back * frame)

    if cv2.waitKey(33) > 0:
        break

capture.release()
cv2.destroyAllWindows()

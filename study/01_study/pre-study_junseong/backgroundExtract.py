import cv2
capture = cv2.VideoCapture(0) 
fgbg = cv2.createBackgroundSubtractorMOG2()
while True:

    # Ensure camera is connected
    if capture.isOpened():
        (status, frame) = capture.read()
            # Ensure valid frame
        if status:
            fgmask = fgbg.apply(frame)
            back=fgbg.getBackgroundImage()

            cv2.imshow('background', back)
            cv2.moveWindow('background', 420 , 500)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
while True:
    ret, cam = cap.read()
    cam = cv2.resize(cam, (250, 250))
    #cam1 = cv2.resize(cam, (250, 250))
    cam2 = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
    #ret, cam2 = cv2.threshold(cam2, 50, 255, cv2.THRESH_BINARY) #THRESH_BINARY 적용

    #ret, cam3 = cv2.threshold(cam2, 50, 255, cv2.THRESH_BINARY_INV) #THRESH_BINARY_INV 적용

    #ret, cam3 = cv2.threshold(cam2, 50, 255, cv2.THRESH_TRUNC) #THRESH_TRUNC 적용

    ret, cam3 = cv2.threshold(cam2, 50, 255, cv2.THRESH_TOZERO) #THRESH_TOZERO 적용

    #ret, cam3 = cv2.threshold(cam2, 50, 255, cv2.THRESH_TOZERO_INV)

    cam3 = cv2.cvtColor(cam3, cv2.COLOR_GRAY2BGR)

    cam_blur = cv2.blur(cam,(5,5))
    cam_blur_gaussian = cv2.GaussianBlur(cam,(5,5),0)

    numpy_horizontal1 = np.hstack((cam, cam3))
    numpy_horizontal2 = np.hstack((cam_blur, cam_blur_gaussian))

    numpy_vertical = np.vstack((numpy_horizontal1, numpy_horizontal2))

    numpy_final = np.vstack((numpy_vertical, numpy_horizontal1))

    if ret:
        #cv2.imshow('cam', cam)
        cv2.imshow('cam', numpy_final)

        if cv2.waitKey(1) &  0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()

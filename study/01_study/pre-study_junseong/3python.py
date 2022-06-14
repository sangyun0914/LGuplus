import cv2
import numpy as np

def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 12, 54)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=10)
    img_erode = cv2.erode(img_dilate, kernel, iterations=8)
    return img_erode

cap = cv2.VideoCapture(0)

while True:
  ret,frame = cap.read()

  contours, _ = cv2.findContours(process(frame), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
  cv2.imshow("Image", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
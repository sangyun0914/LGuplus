import cv2
import numpy as np

MHI_DURATION_FIRST = 15
COLOR = []
for result_idx in range(1,MHI_DURATION_FIRST+1):
    ratio = (result_idx-1)/(MHI_DURATION_FIRST-1)
    print((0,int(255*ratio), int(255*(1-ratio))))
    COLOR.append((0,int(255*ratio), int(255*(1-ratio))))

MHI_DURATION_SECOND = 15
for result_idx in range(1,MHI_DURATION_SECOND+1):
    ratio = (result_idx-1)/(MHI_DURATION_SECOND-1)
    print((int(255*ratio), int(255*(1-ratio)), 0))
    COLOR.append((int(255*ratio), int(255*(1-ratio)), 0))

print(COLOR)
COLOR.reverse()
print(COLOR)
for idx in range(30):

    img = np.zeros((480,640,3) , dtype=np.uint8)

    #속이 채워진 사각형
    cv2.rectangle(img, (300,100), (400,300), COLOR[idx], cv2.FILLED)

    cv2.imshow('img',img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
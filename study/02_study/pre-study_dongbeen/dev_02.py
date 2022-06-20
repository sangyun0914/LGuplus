import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

def overlap(rect1, rect2):
    return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or rect1[1] > rect2[3] or rect1[3] < rect2[1])

def main():
    cap = cv2.VideoCapture('test.avi')
    
    #HOG 보행자 검출 알고리즘 사용
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    x0 = 400
    y0 = 200
    w0 = 150
    h0 = 150
    
    
    while True:
        ret, frame = cap.read()
    
        if not ret:
            break
        
        roi = frame[y0:y0+h0, x0:x0+w0]    
        cv2.rectangle(roi, (0,0), (h0-1, w0-1), (0,255,255))
        
        # 매 프레임마다 보행자 검출
        detected, _ = hog.detectMultiScale(frame) # 사각형 정보를 받아옴
        
        # 검출 결과 화면 표시
        for (x, y, w, h) in detected:
            #c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            #cv2.rectangle(frame, (x, y, w, h), c, 3)
            #if x x0
            rect1 = [x0, y0, x0+w0, y0+h0]
            rect2 = [x, y, x+w, y+h]
            
            if overlap(rect1, rect2):
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
            else:
               cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0)) 
            

            
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) == 27:
            break
     
    #cascade 사용
    
    '''
    while True:
        ret, frame = cap.read()
        body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
        #HOGCascade = cv2.HOGDescriptor('hogcascade_pedestrians.xml')
        #HOGCascade.setSVMDetector(cv2.#HOGDescriptor_getDefaultPeopleDetector())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (300,300))
        #(body, weights) = HOGCascade.detectMultiScale(gray, winStride=winStride,
                                            padding=padding,
                                            scale=scale,
                                            useMeanshiftGrouping=meanshift)

        body = body_cascade.detectMultiScale(frame, 1.01, 2, 0, minSize=(70,70))
        
        for (x,y,w,h) in body:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) == 27:
            break'''           

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
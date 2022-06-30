import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import time

def overlap(rect1, rect2):
    return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or rect1[1] > rect2[3] or rect1[3] < rect2[1])

def main():
    #yolo v4
    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4
    COLORS = [(0,255,255), (255,255,0), (0,255,0), (255,0,0)]

    class_name = []
    with open("classes.txt", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    #net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416,416), scale=1/255, swapRB=True)
    
    cap = cv2.VideoCapture('test.avi')
    
    '''#HOG 보행자 검출 알고리즘 사용
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())'''
    
    x0 = 400
    y0 = 200
    w0 = 150
    h0 = 150

    
    while True:
        ret, frame = cap.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        if not ret:
            break
        
        roi = frame[y0:y0+h0, x0:x0+w0]    
        cv2.rectangle(roi, (0,0), (h0-1, w0-1), (0,255,255))
        
        # 매 프레임마다 보행자 검출
        #detected, _ = hog.detectMultiScale(frame) # 사각형 정보를 받아옴
        
        #yolov4
        start = time.time()
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        end = time.time()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes):
            # classid 0 = human, 사람일 떄만 작동하고, score가 0.58일 때만 사람으로 인식
            if classid == 0 and score >= 0.58:
                color = COLORS[int(classid) % len(COLORS)]
                label = "%s : %f" % (class_names[classid], score)
                rect1 = [x0, y0, x0+w0, y0+h0]
                rect2 = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
                #print(box)
                #print(rect1)
                # 관심구역, 사람 detect 구역 두 구역이 겹치면 침입으로 인식
                if overlap(rect1, rect2):
                   # print('hello')
                    cv2.rectangle(frame,box,(0,0,255), 2)
                    cv2.putText(frame, 'warning', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                else:
                    #print('bye')
                    cv2.rectangle(frame,box,(0,255,0),2) 
                    cv2.putText(frame, 'safe', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        end_drawing = time.time()

        #속도 print
        fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1/(end-start), (end_drawing - start_drawing) * 1000)
        cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
        
        # 검출 결과 화면 표시
        '''for (x, y, w, h) in detected:
            #c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            #cv2.rectangle(frame, (x, y, w, h), c, 3)
            #if x x0
            rect1 = [x0, y0, x0+w0, y0+h0]
            rect2 = [x, y, x+w, y+h]
            
            if overlap(rect1, rect2):
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
            else:
               cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0)) '''
            

            
        
        #frame = cv2.resize(frame, (300,300))
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
import multiprocessing as mp
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import time


def justcam():
    capture = cv2.VideoCapture(0) 
    
    while True:

        # Ensure camera is connected
        if capture.isOpened():
            (status, frame) = capture.read()
            
            # Ensure valid frame
            if status:
                frame=cv2.resize(frame, (0, 0), None, .33, .33)
                cv2.imshow('webcam', frame)
                cv2.moveWindow('webcam', 0 , -100)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()

def faceblur():
    capture = cv2.VideoCapture(0) 
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    while True:
        # Ensure camera is connected
        if capture.isOpened():
            (status, frame) = capture.read()
            
            # Ensure valid frame
            if status:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                for (x, y, w, h) in faces:
                    rate=10
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    roi = frame[y:y+h, x:x+w]   # 관심영역 지정
                    roi = cv2.resize(roi, (w//rate, h//rate)) # 1/rate 비율로 축소
                    # 원래 크기로 확대
                    roi = cv2.resize(roi, (w,h), interpolation=cv2.INTER_AREA)  
                    frame[y:y+h, x:x+w] = roi   # 원본 이미지에 적용
                frame=cv2.resize(frame, (0, 0), None, .33, .33)
                cv2.imshow('face blur', frame)
                cv2.moveWindow('face blur', 420 , -100)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()

def yolo():
    capture = cv2.VideoCapture(0) 
    while True:
        # Ensure camera is connected
        if capture.isOpened():
            (status, frame) = capture.read()
            # bbox, label, conf = cv.detect_common_objects(frame,confidence=0.1, nms_thresh=0.3, model='yolov3-tiny', enable_gpu=False)
            # out = draw_bbox(frame, bbox, label, conf, write_conf=True)
            bbox, label, conf = cv.detect_common_objects(frame)
            out = draw_bbox(frame, bbox, label, conf, write_conf=True)
            # Ensure valid frame
            if status:
                frame=cv2.resize(frame, (0, 0), None, .33, .33)
                cv2.imshow('yolo', frame)
                cv2.moveWindow('yolo', 840 , -100)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()

def edge():
    capture = cv2.VideoCapture(0) 

    while True:

        # Ensure camera is connected
        if capture.isOpened():
            (status, frame) = capture.read()
            frame=cv2.Canny(frame,50,100)
            # Ensure valid frame
            if status:
                frame=cv2.resize(frame, (0, 0), None, .33, .33)
                frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
                cv2.imshow('canny edge', frame)
                cv2.moveWindow('canny edge', 0 , 220)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def blur():
    capture = cv2.VideoCapture(0) 

    while True:

        # Ensure camera is connected
        if capture.isOpened():
            (status, frame) = capture.read()
            frame = cv2.blur(frame,(5,5))
            # Ensure valid frame
            if status:
                frame=cv2.resize(frame, (0, 0), None, .33, .33)
                cv2.imshow('blur', frame)
                cv2.moveWindow('blur', 420 , 220)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()

def back():
    capture = cv2.VideoCapture(0) 
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while True:

        # Ensure camera is connected
        if capture.isOpened():
            (status, frame) = capture.read()
            # Ensure valid frame
            if status:
                frame=cv2.resize(frame, (0, 0), None, .33, .33)
                fgmask = fgbg.apply(frame)
                cv2.imshow('background', fgmask)
                cv2.moveWindow('background', 840 , 220)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()

def threhold():
    capture = cv2.VideoCapture(0) 
    while True:

        # Ensure camera is connected
        if capture.isOpened():
            (status, frame) = capture.read()
            frame=cv2.medianBlur(frame,5)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
            # Ensure valid frame
            if status:
                frame=cv2.resize(frame, (0, 0), None, .33, .33)
                cv2.imshow('threhold', frame)
                cv2.moveWindow('threhold', 0 , 500)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    th1 = mp.Process(target=justcam, args=())
    th2 = mp.Process(target=faceblur, args=())
    th3 = mp.Process(target=yolo, args=())
    th4 = mp.Process(target=edge, args=())
    th5 = mp.Process(target=blur, args=())
    th6 = mp.Process(target=back, args=())
    th7 = mp.Process(target=threhold, args=())

    th1.start()
    th2.start()
    th3.start()
    th4.start()
    th5.start()
    th6.start()
    th7.start()
    
    

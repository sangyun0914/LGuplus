import multiprocessing as mp
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import timeit

#웹캠만 출력
def justcam():
    capture = cv2.VideoCapture(0)
    while True:

        # Ensure camera is connected
        if capture.isOpened():
            start_t = timeit.default_timer()
            (status, frame) = capture.read()
            # Ensure valid frame
            if status:
                terminate_t = timeit.default_timer()
                FPS = int(1./(terminate_t - start_t ))
                FPS=str(FPS)
                cv2.putText(frame, "FPS : "+FPS,(20, 60), 0, 2, (255,0,0),3)
                frame=cv2.resize(frame, (0, 0), None, .33, .33)
                cv2.imshow('webcam', frame)
                cv2.moveWindow('webcam', 0 , -100)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()

#얼굴 감지 후, 블러처리
def faceblur():
    capture = cv2.VideoCapture(0) 
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    while True:
        # Ensure camera is connected
        if capture.isOpened():
            start_t = timeit.default_timer()
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
                terminate_t = timeit.default_timer()
                FPS = int(1./(terminate_t - start_t ))
                FPS=str(FPS)
                cv2.putText(frame, "FPS : "+FPS,(20, 60), 0, 2, (255,0,0),3)
                frame=cv2.resize(frame, (0, 0), None, .33, .33)
                cv2.imshow('face blur', frame)
                cv2.moveWindow('face blur', 420 , -100)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()

#yolo를 사용한 물체 인식
def yolo():
    capture = cv2.VideoCapture(0) 
    while True:
        # Ensure camera is connected
        if capture.isOpened():
            start_t = timeit.default_timer()
            (status, frame) = capture.read()
            # bbox, label, conf = cv.detect_common_objects(frame,confidence=0.1, nms_thresh=0.3, model='yolov3-tiny', enable_gpu=False)
            # out = draw_bbox(frame, bbox, label, conf, write_conf=True)
            bbox, label, conf = cv.detect_common_objects(frame)
            out = draw_bbox(frame, bbox, label, conf, write_conf=True)
            # Ensure valid frame
            if status:
                terminate_t = timeit.default_timer()
                FPS = round((1./(terminate_t - start_t )),4)
                FPS=str(FPS)
                cv2.putText(frame, "FPS : "+FPS,(20, 60), 0, 2, (255,0,0),3)
                frame=cv2.resize(frame, (0, 0), None, .33, .33)
                cv2.imshow('yolo', frame)
                cv2.moveWindow('yolo', 840 , -100)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()

#canny edge 감지
def edge():
    capture = cv2.VideoCapture(0) 

    while True:

        # Ensure camera is connected
        if capture.isOpened():
            start_t = timeit.default_timer()
            (status, frame) = capture.read()
            frame=cv2.Canny(frame,50,100)
            # Ensure valid frame
            if status:
                #frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
                terminate_t = timeit.default_timer()
                FPS = int(1./(terminate_t - start_t ))
                FPS=str(FPS)
                cv2.putText(frame, "FPS : "+FPS,(20, 60), 0, 2, (255,0,0),3)
                frame=cv2.resize(frame, (0, 0), None, .33, .33)
                cv2.imshow('canny edge', frame)
                cv2.moveWindow('canny edge', 0 , 220)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

#블러처리
def blur():
    capture = cv2.VideoCapture(0) 

    while True:

        # Ensure camera is connected
        if capture.isOpened():
            start_t = timeit.default_timer()
            (status, frame) = capture.read()
            frame = cv2.blur(frame,(5,5))
            # Ensure valid frame
            if status:
                terminate_t = timeit.default_timer()
                FPS = int(1./(terminate_t - start_t ))
                FPS=str(FPS)
                cv2.putText(frame, "FPS : "+FPS,(20, 60), 0, 2, (255,0,0),3)
                frame=cv2.resize(frame, (0, 0), None, .33, .33)
                cv2.imshow('blur', frame)
                cv2.moveWindow('blur', 420 , 220)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()

#배경 감지
def backsub():
    capture = cv2.VideoCapture(0) 
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while True:

        # Ensure camera is connected
        if capture.isOpened():
            start_t = timeit.default_timer()
            (status, frame) = capture.read()
            # Ensure valid frame
            if status:
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fgmask = fgbg.apply(frame)
                terminate_t = timeit.default_timer()
                FPS = int(1./(terminate_t - start_t ))
                FPS=str(FPS)
                cv2.putText(fgmask, "FPS : "+FPS,(20, 60), 0, 2, (255,0,0),3)
                fgmask=cv2.resize(fgmask, (0, 0), None, .33, .33)
                cv2.imshow('backgroundsub', fgmask)
                cv2.moveWindow('backgroundsub', 840 , 220)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()

#배경만
def back():
    capture = cv2.VideoCapture(0) 
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while True:

        # Ensure camera is connected
        if capture.isOpened():
            start_t = timeit.default_timer()
            (status, frame) = capture.read()
            # Ensure valid frame
            if status:
                fgmask = fgbg.apply(frame)
                back=fgbg.getBackgroundImage()
                terminate_t = timeit.default_timer()
                FPS = int(1./(terminate_t - start_t ))
                FPS=str(FPS)
                cv2.putText(back, "FPS : "+FPS,(20, 60), 0, 2, (255,0,0),3)
                back=cv2.resize(back, (0, 0), None, .33, .33)
                cv2.imshow('background', back)
                cv2.moveWindow('background', 420 , 500)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()

#threhold 적용
def threhold():
    capture = cv2.VideoCapture(0) 
    while True:

        # Ensure camera is connected
        if capture.isOpened():
            start_t = timeit.default_timer()
            (status, frame) = capture.read()
            frame=cv2.medianBlur(frame,5)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
            # Ensure valid frame
            if status:
                terminate_t = timeit.default_timer()
                FPS = int(1./(terminate_t - start_t ))
                FPS=str(FPS)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.putText(frame, "FPS : "+FPS,(20, 60), 0, 2, (255,0,0),3)
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
    #멀티 프로세싱
    th1 = mp.Process(target=justcam, args=())
    th2 = mp.Process(target=faceblur, args=())
    th3 = mp.Process(target=yolo, args=())
    th4 = mp.Process(target=edge, args=())
    th5 = mp.Process(target=blur, args=())
    th6 = mp.Process(target=backsub, args=())
    th7 = mp.Process(target=threhold, args=())
    th8 = mp.Process(target=back, args=())

    th1.start()
    th2.start()
    th3.start()
    th4.start()
    th5.start()
    th6.start()
    th7.start()
    th8.start()
    
    

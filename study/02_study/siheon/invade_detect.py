import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

capture = cv2.VideoCapture("video.mp4")
(status, frame) = capture.read()
lx,ly,w,h=cv2.selectROI("location",frame,False)
cv2.destroyWindow("location");   
cnt=0
while True:

    # Ensure camera is connected
    if capture.isOpened():
        (status, frame) = capture.read()
        frame=cv2.rectangle(frame,(lx,ly),(lx+w,ly+h),(255,0,0),1)
        # Ensure valid frame
        if status:
            bbox, label, conf = cv.detect_common_objects(frame,confidence=0.1, nms_thresh=0.3, model='yolov3-tiny', enable_gpu=False)
            out = draw_bbox(frame, bbox, label, conf, write_conf=True)

            if(bbox):
                k=0
                for obj in bbox:
                    if(not(obj[0]>lx+w or obj[1]>ly+h or obj[2]<lx or obj[3]<ly)and label[k]=="person"):
                        cnt+=1
                    k+=1
                if(cnt!=0):
                    scnt=str(cnt)
                    cv2.putText(frame, scnt+"invade!",(20, 100), 0, 3, (255,0,0),3)
            cv2.imshow('invade detect', frame)
            cnt=0
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
capture.release()
cv2.destroyAllWindows()
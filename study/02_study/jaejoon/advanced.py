import cv2
import numpy as np

video = cv2.VideoCapture('test2.mp4')
mask = cv2.bgsegm.createBackgroundSubtractorMOG()

# roi 지정
_, frame = video.read()
x, y, w, h = cv2.selectROI('video', frame, False)
cv2.destroyWindow("video")

# yolo 모델 불러오기
net = cv2.dnn.readNet("yolov3.weights",
                      "yolov3.cfg")  # YOLO 파일 불러와서 모델 만들기
classes = []
with open("coco.names", "r") as f:  # 인식가능한 오브젝트(클래스) 이름들이 저장되어있는 파일
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 전경 라벨링 바운딩 박스 부분을 yolo로 돌려서 사람인지 인식

cnt = 0


def isHuman(frame):
    flag = False

    frame = cv2.resize(frame, dsize=(0, 0), fx=4, fy=4,
                       interpolation=cv2.INTER_AREA)
    frame = cv2.GaussianBlur(frame, (15, 15), 0)

    cv2.imwrite('img/'+str(cnt)+'.jpeg', frame)

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # 0.5보다 높으면 인식
                #print('yolo detected')
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                if class_id == 0:
                    flag = True
                    return flag

    return flag


while(True):
    ret, frame = video.read()
    if w and h:
        roi = frame[y:y+h, x:x+w]

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # roi 부분만 전경 추출 후 라벨링
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    back = mask.apply(blur)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(back)

    flag = False  # 첫번째 stats 무시하려는 플래그
    detect = False  # roi에 사람 인식되었는지 체크

    for s in stats:
        if flag:
            #cv2.rectangle(roi, (s[0], s[1]), (s[0] + s[2], s[1]+s[3]), (255, 0, 0), 1)
            #cv2.imshow('maybe', roi[s[0]:s[0]+s[2], s[1]:s[1]+s[3]])
            #cv2.imwrite('img/'+str(cnt)+'.jpeg',roi[s[1]:s[1]+s[3], s[0]:s[0]+s[2]])
            cnt += 1
            if (s[4] > 100):
                if isHuman(roi[s[1]:s[1]+s[3], s[0]:s[0]+s[2]]):
                    # cv2.imshow('maybe human', roi[s[0]:s[0]+s[2], s[1]:s[1]+s[3]])
                    detect = True
                    print('human detected')
        else:
            flag = True

    if detect:
        cv2.rectangle(frame, (0, 0),
                      (frame.shape[1], frame.shape[0]), (0, 0, 255), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) > 0:
        break

video.release()
cv2.destroyAllWindows()

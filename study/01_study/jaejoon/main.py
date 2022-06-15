import cv2
import numpy as np

# 웹캠 설정
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 1. 처음에 웹캠을 키고 배경 영상을 찍음 
# 새로운 프레임이 들어오면 그 프레임에 배경영상을 빼서 차이가 일정 수준이 넘어가면 전경이라고 인식
# 2. 자체 내장 함수 사용 
# 픽셀 값이 일정 시간 동안 변화가 없으면 배경이라고 인식

# 배경 영상 등록
ret, back1 = capture.read()
    
# 연산 속도를 높이기 위해 그레이스케일 영상으로 변환
back1 = cv2.cvtColor(back1, cv2.COLOR_BGR2GRAY)

# 배경 추출 알고리즘 함수, 오랜 시간 동안 변하지 않는 픽셀들을 배경으로 판단하는 방식
mask = cv2.bgsegm.createBackgroundSubtractorMOG()

# Load Yolo
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

while(True):
    ret, frame = capture.read()
    height, width, channels = frame.shape

    # 현재 프레임 영상 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 배경 영상과 현재 프레임의 차이 절댓값 구함
    diff = cv2.absdiff(gray, back1)

    # 차이가 40이상이면 전경이라고 판단
    # diff2는 전경으로 추출한 부분을 보여주기 위한 것
    # filter1은 원래 프레임에 곱해서 전경 부분만 컬러로 뽑아내기 위한 것
    _, diff2 = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    diff2 = cv2.cvtColor(diff2, cv2.COLOR_GRAY2BGR)
    _, filter1 = cv2.threshold(diff, 40, 1, cv2.THRESH_BINARY)

    # 원래 프레임에 곱하기 위해 필터를 3채널로 만듦
    filter2 = cv2.merge((filter1, filter1, filter1))

    # 전경 추출, 흑백
    back2 = mask.apply(frame)
    back2 = cv2.cvtColor(back2, cv2.COLOR_GRAY2BGR)

    # 원본 영상에서 전경 추출을 위해 필터 생성
    _, filter3 = cv2.threshold(back2, 30, 1, cv2.THRESH_BINARY)

    # 영상 여러개 한번에 보여주기 위해 합치는 과정
    numpy_horizontal = np.hstack((diff2, frame * filter2, back2, frame * filter3))

    # 합친 화면 띄워줌
    cv2.imshow('background', numpy_horizontal)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 흑백으로 변환

    # threshold    
    ret, th = cv2.threshold(gray ,127, 255, cv2.THRESH_BINARY) # 임계값(127) 넘으면 백, 아니면 흑으로 바꿈
    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2) # 이미지 영역별로 이진화 진행, 영역의 평균값을 임계값으로 설정, 영역(블록)사이즈 = 15

    # blur
    blur = cv2.blur(frame, (5, 5)) # 5 * 5 박스내 픽셀 평균값
    gblur = cv2.GaussianBlur(frame, (15,15), 0) # 2차원 가우시안 분포 이용
    mblur = cv2.medianBlur(frame, 7) # 박스 내 픽셀들의 중간값, 박스 사이즈 = 7 
    bfblur = cv2.bilateralFilter(frame, 10, 75, 75) # 윤곽선은 살리고 나머지는 블러 처리

    # canny edge detection
    canny = cv2.Canny(gray, 70, 200)

    # face detection
    ret, mosaic = capture.read()
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(mosaic)

    # mosaic
    for (x, y, w, h) in faces:
        roi = mosaic[y:y+h, x:x+w]   # 관심영역 지정
        roi = cv2.resize(roi, (w//15, h//15)) # 1/20 비율로 축소
        # 원래 크기로 확대
        roi = cv2.resize(roi, (w,h), interpolation=cv2.INTER_AREA)  
        mosaic[y:y+h, x:x+w] = roi # 원본 이미지에 적용

    result1 = np.hstack((th, th2, canny))
    result2 = np.hstack((frame, gblur, mosaic))
    cv2.imshow('1', result1)
    cv2.imshow('2', result2)

    # YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow("YOLO", frame)

    # 키보드 입력시 종료
    if cv2.waitKey(33) > 0:
        break

# 웹캠 종료, 모든 창 종료
capture.release()
cv2.destroyAllWindows()
 
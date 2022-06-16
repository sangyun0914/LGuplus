# https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html#scales-for-handling-different-sizes
# https://thomapple.tistory.com/entry/YOLO-%EC%82%AC%EB%AC%BC%EC%9D%B8%EC%8B%9D-python

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
# 가우시안 분산값 K(K=3~5)의 홉합에 의해 각 배경 픽셀을 구성하는 방법
# 홉합의 가중치는 장면에서 이들 색상값들이 머무르고 있는 시간 비율
# 배경으로써 판단될 수 있는 색상은 더 오랜 시간동안 변하지 않는 것
mask = cv2.bgsegm.createBackgroundSubtractorMOG()

# YOLO 설정
net = cv2.dnn.readNet("yolov3-tiny.weights",
                      "yolov3-tiny.cfg")  # YOLO 파일 불러와서 모델 만들기
classes = []
with open("coco.names", "r") as f:  # 인식가능한 오브젝트(클래스) 이름들이 저장되어있는 파일
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
np.random.seed(42)
colors = np.random.randint(0, 255, size=(
    len(classes), 3), dtype='uint8')  # 클래스 별 랜덤 색상 정해줌

while(True):
    ret, frame1 = capture.read()

    # 현재 프레임 영상 그레이스케일 변환
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # 배경 영상과 현재 프레임의 차이 절댓값 구함
    diff = cv2.absdiff(gray, back1)

    # 차이가 40이상이면 전경이라고 판단
    # diff2는 전경으로 추출한 부분을 보여주기 위한 것
    # filter1은 원래 프레임에 곱해서 전경 부분만 컬러로 뽑아내기 위한 것
    _, diff2 = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    diff2 = cv2.cvtColor(diff2, cv2.COLOR_GRAY2BGR)
    _, filter1 = cv2.threshold(diff, 40, 1, cv2.THRESH_BINARY)

    # 원래 프레임에 곱하기 위해 필터를 3채널로 만들었음
    filter2 = cv2.merge((filter1, filter1, filter1))

    # 아래는 BackgroundSubtractorMOG 사용한 방식
    # 전경 추출, 흑백
    back2 = mask.apply(frame1)
    back2 = cv2.cvtColor(back2, cv2.COLOR_GRAY2BGR)

    # 원본 영상에서 전경 추출을 위해 필터 생성
    _, filter3 = cv2.threshold(back2, 30, 1, cv2.THRESH_BINARY)

    ###################################################################################

    ret, frame2 = capture.read()

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)  # 흑백으로 변환

    # threshold
    # 임계값(127) 넘으면 백, 아니면 흑으로 바꿈
    ret, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 이미지 영역별로 이진화 진행, 영역의 평균값을 임계값으로 설정, 영역(블록)사이즈 = 15
    th2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)

    # blur
    blur = cv2.blur(frame2, (5, 5))  # 5 * 5 박스내 픽셀 평균값
    # 2차원 가우시안 분포 이용, 가까운 픽셀에서 영향 많이 받음
    gblur = cv2.GaussianBlur(frame2, (15, 15), 0)
    mblur = cv2.medianBlur(frame2, 7)  # 박스 내 픽셀들의 중간값, 박스 사이즈 = 7
    bfblur = cv2.bilateralFilter(frame2, 10, 75, 75)  # 윤곽선은 살리고 나머지는 블러 처리

    # canny edge detection
    # https://blog.naver.com/samsjang/220507996391
    # sobel 커널을 적용하여 gradient 계산한 후, 최댓값이 아닌 것들은 0으로 만듦
    # minval보다 낮으면 엣지 아니라고 판단. maxval보다 높으면 무조건 엣지라고 판단, 사잇값이면 픽셀 연결구조 보고 판단
    canny = cv2.Canny(gray, 70, 200)

    # face detection
    ret, mosaic = capture.read()
    face_classifier = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(mosaic)

    # mosaic
    for (x, y, w, h) in faces:
        roi = mosaic[y:y+h, x:x+w]   # 관심영역 지정
        roi = cv2.resize(roi, (w//15, h//15))  # 1/20 비율로 축소
        # 원래 크기로 확대
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
        mosaic[y:y+h, x:x+w] = roi  # 원본 이미지에 적용

    th = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    th2 = cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR)
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    ###################################################################################

    ret, frame3 = capture.read()
    height, width, channels = frame3.shape

    # YOLO 객체 탐지
    blob = cv2.dnn.blobFromImage(
        frame3, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # 네트워크의 인풋은 blob object임
    # A blob is a 4D numpy array object (images, channels, width, height).
    # cv.dnn.blobFromImage(img, scale, size, mean)
    # 아래는 각 파라미터 설명
    # the image to transform
    # the scale factor (1/255 to scale the pixel values to [0..1])
    # the size, here a 416x416 square image
    # the mean value (default=0)
    # the option swapBR=True (since OpenCV uses BGR)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # The outputs object are vectors of lenght 85
    # 4x the bounding box (centerx, centery, width, height)
    # 1x box confidence
    # 80x class confidence

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # 0.5보다 높으면 인식
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 중복 박스 제거, Non maximum suppresion
    # https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.g137784ab86_4_2923
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 물체 주변 박스, confidence 그려줌
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(frame3, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
            cv2.putText(frame3, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 영상 여러개 한번에 보여주기 위해 합치는 과정
    result0 = np.hstack((frame1 * filter2, frame1 * filter3))
    result1 = np.hstack((th, th2, canny))
    result2 = np.hstack((gblur, bfblur, mosaic))
    result3 = np.hstack((gblur, bfblur, mosaic))
    result4 = np.hstack((frame1 * filter2, frame1 * filter3, frame3))
    result5 = np.vstack((result1, result3, result4))

    cv2.imshow("pre-study", result5)

    # 키보드 입력시 종료
    if cv2.waitKey(33) > 0:
        break

# 웹캠 종료, 모든 창 종료
capture.release()
cv2.destroyAllWindows()

cv2.im

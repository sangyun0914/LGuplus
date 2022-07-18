import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

video = 'test.avi'
extract = np.empty((1, 88))

cap = cv2.VideoCapture(video)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        temp = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten(
        ) if results.pose_world_landmarks else np.zeros(88)

        # print(temp[44:])

        # 얼굴을 제외한 22개의 랜드마크만 사용
        extract = np.append(extract, [temp[44:]], axis=0)

extract = np.delete(extract, (0), axis=0)
# print(extract.shape)

# 30 프레임에서 추출한 관절 정보들을 하나의 넘파이 배열로 저장
np.save(video, extract)
cap.release()

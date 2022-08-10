import cv2
import test
import mediapipe as mp
import numpy as np
import copy
import torch
import torch.nn as nn
import extraFeatures

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

seq_length = 20  # 20 프레임
data_dim = 96  # 22개의 랜드마크, 랜드마크의 x, y, z, visibility + 8개 관절 각도


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=layers, batch_first=True, bias=True, dropout=0.3, bidirectional=False)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=True)
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view([-1, seq_length, data_dim])
        x, _status = self.lstm(x)
        x = self.silu(self.fc1(x[:, -1]))
        x = self.silu(self.fc2(x))
        x = self.fc3(x)
        # x = self.softmax(x)
        return x


model = test.initModel()

extract = np.empty((1, 96))

cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # 미디어파이프를 이용하여 스켈레톤 추출
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

        temp = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten(
        ) if results.pose_world_landmarks else np.zeros(132)

        # 얼굴을 제외한 22개의 랜드마크만 사용하기 위해 0~43번 인덱스 내용은 버림
        angles = extraFeatures.extractAngles(results)
        temp2 = np.append(temp[44:], angles)
        extract = np.append(extract, [temp2], axis=0)
        extract = extract.astype(np.float32)

        image = cv2.flip(image, 1)

        if(extract.shape[0] > seq_length):
            extract = np.delete(extract, (0), axis=0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            prob, action = test.testModel(model, torch.Tensor(extract))
            prob = prob.item()
            cv2.putText(image, action, (50, 100),
                        font, 2, (255, 0, 0), 2)
            cv2.putText(image, str(prob), (50, 300),
                        font, 2, (255, 0, 0), 2)

        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) > 0:
            break


cap.release()

import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
import copy
import time

mp_pose = mp.solutions.pose

actions = ['squat-down', 'squat-up', 'lunge-down', 'lunge-up']

video = '/Users/jaejoon/LGuplus/main_project/lstm/squatdown/01-202207081454-1-20.avi'


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
        x = x.view([-1, 30, 88])
        x, _status = self.lstm(x)
        x = self.relu(self.fc1(x[:, -1]))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def extractPose(video_path):
    # 프레임마다 뽑힌 스켈레톤 좌표를 하나로 모으기 위하여 비어있는 넘파이 배열 생성
    extract = np.empty((1, 88))

    # 비디오 캡쳐 시작
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # 미디어파이프를 이용하여 스켈레톤 추출
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            temp = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten(
            ) if results.pose_world_landmarks else np.zeros(132)

            # 얼굴을 제외한 22개의 랜드마크만 사용하기 위해 0~43번 인덱스 내용은 버림
            temp2 = copy.deepcopy(temp[44:])
            extract = np.append(extract, [temp2], axis=0)

    # 첫번째 열은 아무 의미 없는 값이 들어가있기 때문에 지워줌
    extract = np.delete(extract, (0), axis=0)
    extract = extract.astype(np.float32)

    cap.release()
    return torch.Tensor(extract)


def initModel():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load('./model/model_mk1.pt', map_location=device)
    print(model)
    return model


def testModel(model, test_data):
    start = time.time()
    model.eval()
    with torch.no_grad():
        out = model(test_data)
        print(actions[out.numpy().argmax()])
    m, s = divmod(time.time() - start, 60)
    print(f'Inference time: {m:.0f}m {s:.5f}s')
    print(out)
    return actions[out.numpy().argmax()]


def main():
    test_data = extractPose(video)
    model = initModel()
    testModel(model, test_data)


if __name__ == '__main__':
    main()

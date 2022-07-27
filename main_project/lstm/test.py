import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mediapipe as mp
import numpy as np
import copy
import time


mp_pose = mp.solutions.pose

actions = ['squat-down', 'squat-up', 'pushup-down',
           'pushup-up', 'lunge-down', 'lunge-up']

seq_length = 30  # 20 프레임
data_dim = 88  # 22개의 랜드마크, 랜드마크의 x, y, z, visibility + 8개 관절 각도


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


def initModel():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load('./model/model_mk3.pt', map_location=device)
    print(model)
    return model


def testModel(model, test_data):
    start = time.time()
    model.eval()
    with torch.no_grad():
        out = model(test_data)
        out = np.squeeze(out)
        out = F.softmax(out, dim=0)
        # print(actions[out.numpy().argmax()])
    m, s = divmod(time.time() - start, 60)
    print(f'Inference time: {m:.0f}m {s:.5f}s')
    # print(out)
    # print(out.numpy().argmax())
    return out[out.numpy().argmax()], actions[out.numpy().argmax()]


def main():
    return 0


if __name__ == '__main__':
    main()

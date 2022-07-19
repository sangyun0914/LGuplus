import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

seq_length = 30  # 30 프레임
data_dim = 88  # 22개의 랜드마크, 랜드마크의 x, y, z, visibility
hidden_dim = 30
output_dim = 2  # 스쿼트 업, 다운
learning_rate = 0.01
epochs = 1000


class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.label = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        sample


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)
        self.softmax = nn.Softmax(output_dim)

    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


net = Net(data_dim, hidden_dim, output_dim, 1)

# loss & optimizer setting
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

for i in range(epochs):
    optimizer.zero_grad()
    outputs = net(trainX_tensor)
    loss = criterion(outputs, trainY_tensor)
    loss.backward()
    optimizer.step()
    print(i, loss.item())

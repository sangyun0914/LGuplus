import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import time

actions = ['squat-down', 'squat-up', 'pushup-down',
           'pushup-up', 'lunge-down', 'lunge-up']

seq_length = 30  # 30 프레임
data_dim = 88  # 22개의 랜드마크, 랜드마크의 x, y, z, visibility
hidden_dim = 20
output_dim = len(actions)  # 운동 종류
lstm_layers = 3
learning_rate = 0.0005
epochs = 100
batch_size = 32
model_name = 'model_mk3'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyDataset(Dataset):
    def __init__(self):
        data = np.loadtxt('mydataset.csv', delimiter=",", dtype=np.float32)
        print(data.shape)
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, 0:-output_dim])
        self.y_data = torch.from_numpy(data[:, -output_dim:])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = MyDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


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


model = Model(data_dim, hidden_dim, output_dim, lstm_layers)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = target.long()
        target = target.argmax(axis=1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {:4d} | Batch Status: {:3d}/{} ({:2.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


if __name__ == '__main__':

    # print(pytorch_model_summary.summary(
    #    model, torch.zeros(249, 2640), show_input=False))
    since = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60)
        # print(f'Training time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')

    torch.save(model, './model/{}.pt'.format(model_name))
    print(model_name, 'Saved!')

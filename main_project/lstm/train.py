from statistics import mode
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import time

seq_length = 30  # 30 프레임
data_dim = 88  # 22개의 랜드마크, 랜드마크의 x, y, z, visibility
hidden_dim = 20
output_dim = 2  # 스쿼트 업, 다운
lstm_layers = 2
learning_rate = 0.001
epochs = 500
batch_size = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyDataset(Dataset):
    def __init__(self):
        data = np.loadtxt('mydataset.csv', delimiter=",", dtype=np.float32)
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, 0:-2])
        self.y_data = torch.from_numpy(data[:, -2:])

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
                            num_layers=layers, batch_first=True, bias=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view([-1, 30, 88])
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        x = self.softmax(x)
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
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


if __name__ == '__main__':

    # print(pytorch_model_summary.summary(
    #    model, torch.zeros(249, 2640), show_input=False))

    m = nn.Softmax(dim=1)
    input = torch.randn(3, 2)
    print(input)
    output = m(input)
    print(output)

    since = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60)
        # print(f'Training time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')

    torch.save(model, './model/model_mk0.pt')
    print('Model Saved!')

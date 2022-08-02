from model_v3 import Model
from configs import *
from myDataset import MyDataset

# 모델 파라미터 설정
lstm_layers = config['lstm_layers']
dropout = config['dropout']

# 학습 파라미터 설정
learning_rate = 0.0005
epochs = 100
batch_size = 32
model_name = 'model_mk5'

# cpu, gpu 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 데이터 셋 설정
dataset = MyDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# 모델 설정
model = Model(lstm_layers, dropout)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습

loss_val = []


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
            loss_val.append(loss.item())

        if batch_idx % 10 == 0:
            print('Train Epoch: {:4d} | Batch Status: {:3d}/{} ({:2.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


if __name__ == '__main__':
    since = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60)
        # print(f'Training time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')

    plt.plot(np.array(loss_val))
    plt.show()

    torch.save(model, './model/{}.pt'.format(model_name))
    print(model_name, 'Saved!')

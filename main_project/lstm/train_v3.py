from model_v3 import Model
from configs import *
from myDataset import MyDataset

# 모델 파라미터 설정
lstm_layers = config['lstm_layers']
data_dim = config['data_dim']
dropout = config['dropout']
seq_length = config['seq_length']

# 학습 파라미터 설정
learning_rate = 0.0001
epochs = 500
batch_size = 32
model_name = 'model_mk6'

# cpu, gpu 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 데이터 셋 설정
train_dataset = MyDataset('mydataset_v3_train_30frames.csv')
validation_dataset = MyDataset('mydataset_v3_valid_30frames.csv')
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(
    dataset=validation_dataset, batch_size=batch_size, shuffle=True)

# 모델 설정
model = Model(lstm_layers, data_dim, seq_length, dropout)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습
train_loss_val = []
valid_loss_val = []


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
            print('Train Epoch: {:4d}/{} | Batch Status: {:3d}/{} ({:2.0f}%) | Training Loss: {:.6f}'.format(
                epoch, epochs, batch_idx *
                len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    with torch.no_grad():
        model.eval()
        for batch_idx, (data, target) in enumerate(validation_loader):
            data = data.to(device)
            target = target.to(device)
            target = target.long()
            target = target.argmax(axis=1)
            valid_output = model(data)
            valid_loss = criterion(valid_output, target)

    train_loss_val.append(loss.item())
    valid_loss_val.append(valid_loss.item())

    if (epoch % 50) == 0:
        torch.save(model, './model/{}_{}_{:.4f}_{:.4f}.pt'.format(
            model_name, epoch, loss.item(), valid_loss.item()))


if __name__ == '__main__':
    since = time.time()

    print(pytorch_model_summary.summary(
        model, torch.zeros(1, config['seq_length'], config['data_dim']), show_input=False))

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Training time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')

    plt.plot(np.array(train_loss_val), 'b')
    plt.plot(np.array(valid_loss_val), 'r')
    plt.savefig('loss graph.png')
    plt.show()

    torch.save(model, './model/{}_final.pt'.format(model_name))
    print(model_name, 'Saved!')

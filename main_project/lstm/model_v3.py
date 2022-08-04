from matplotlib.pyplot import axis
from configs import *

seq_length = config['seq_length']
data_dim = config['data_dim']

# 왼쪽 팔, 오른쪽 팔, 왼쪽 다리, 오른쪽 다리 각각 나눠서 처리하는 모델


class Model(nn.Module):
    def __init__(self, layers, dropout):
        super(Model, self).__init__()
        self.lstm1 = nn.LSTM(22, 10,
                             num_layers=layers, batch_first=True, bias=True, dropout=dropout)
        self.lstm2 = nn.LSTM(22, 10,
                             num_layers=layers, batch_first=True, bias=True, dropout=dropout)
        self.lstm3 = nn.LSTM(22, 10,
                             num_layers=layers, batch_first=True, bias=True, dropout=dropout)
        self.lstm4 = nn.LSTM(22, 10,
                             num_layers=layers, batch_first=True, bias=True, dropout=dropout)
        self.lstm5 = nn.LSTM(20, 10,
                             num_layers=layers, batch_first=True, bias=True, dropout=dropout)
        self.lstm6 = nn.LSTM(20, 10,
                             num_layers=layers, batch_first=True, bias=True, dropout=dropout)
        self.lstm7 = nn.LSTM(20, 10,
                             num_layers=layers, batch_first=True, bias=True, dropout=dropout)
        self.fc1 = nn.Linear(10, 10, bias=True)
        self.fc2 = nn.Linear(10, len(actions), bias=True)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = x.view([-1, seq_length, data_dim])
        # print(x.shape)

        left_upper_data = x[:, :, 0:22]
        right_upper_data = x[:, :, 22:44]
        left_lower_data = x[:, :, 44:66]
        right_lower_data = x[:, :, 66:88]

        left_upper, _ = self.lstm1(left_upper_data)
        right_upper, _ = self.lstm2(right_upper_data)
        left_lower, _ = self.lstm3(left_lower_data)
        right_lower, _ = self.lstm4(right_lower_data)

        upper = torch.cat([left_upper, right_upper], dim=2)
        lower = torch.cat([left_lower, right_lower], dim=2)

        upper, _ = self.lstm5(upper)
        lower, _ = self.lstm6(lower)

        body = torch.cat([upper, lower], dim=2)
        body, _ = self.lstm7(body)
        x = self.silu(self.fc1(body[:, -1, :]))
        x = self.fc2(x)
        return x

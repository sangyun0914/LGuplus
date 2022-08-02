from configs import *

# 데이터 셋 csv 파일명 유의!!
dataset_file = 'mydataset_v3.csv'

output_dim = config['output_dim']


class MyDataset(Dataset):
    def __init__(self):
        data = np.loadtxt(dataset_file, delimiter=",", dtype=np.float32)
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, 0:-output_dim])
        self.y_data = torch.from_numpy(data[:, -output_dim:])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

import os
import numpy as np

# 운동 종류 지정
actions = ['squatdown', 'squatup']


def makelabel(action):
    idx = actions.index(action)
    label = np.zeros(len(actions), dtype=np.float32)
    label[idx] = 1.0
    return label


def main():
    dataset = np.empty((1, 2640 + len(actions)))
    for action in actions:
        for filename in os.listdir("./{0}_csv".format(action)):
            file = np.loadtxt(os.path.join(
                "./{0}_csv".format(action), filename), delimiter=",", dtype=np.float32)
            # 30 프레임의 데이터를 하나로 펼침
            file = file.flatten()
            # 데이터 마지막에 라벨링 추가
            file = np.concatenate([file, makelabel(action)])
            dataset = np.append(dataset, [file], axis=0)
    dataset = np.delete(dataset, (0), axis=0)
    # print(dataset.shape)
    np.savetxt('mydataset.csv', dataset, delimiter=",", fmt='%.5f')


if __name__ == '__main__':
    main()

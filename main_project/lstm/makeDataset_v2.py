import os
import numpy as np
import sampling20 as smp

# 운동 종류 지정
actions = ['squat-down', 'squat-up', 'pushup-down',
           'pushup-up', 'lunge-down', 'lunge-up']


def makelabel(action):
    idx = actions.index(action)
    label = np.zeros(len(actions), dtype=np.float32)
    label[idx] = 1.0
    return label


def main():
    dataset = np.empty((1, 1920 + len(actions)))
    for action in actions:
        for filename in os.listdir("./csv_extra/{0}_csv".format(action)):
            print(filename)
            if filename.endswith('.DS_Store'):
                continue
            file = np.loadtxt(os.path.join(
                "./csv_extra/{0}_csv".format(action), filename), delimiter=",", dtype=np.float32)

            # 하나의 비디오를 5번 랜덤 샘플링
            for i in range(5):
                sample_file = smp.sample20(file)
                # 샘플링한 20프레임의 데이터를 하나로 펼침
                sample_file = sample_file.flatten()
                # 데이터 마지막에 라벨링 추가
                sample_file = np.concatenate([sample_file, makelabel(action)])
                dataset = np.append(dataset, [sample_file], axis=0)

    dataset = np.delete(dataset, (0), axis=0)
    print(actions)
    print(dataset.shape)
    np.savetxt('mydataset_v2.csv', dataset, delimiter=",", fmt='%.5f')


if __name__ == '__main__':
    main()

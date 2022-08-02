from configs import *


def makelabel(action):
    idx = actions.index(action)
    label = np.zeros(len(actions), dtype=np.float32)
    label[idx] = 1.0
    return label


def main():
    dataset = np.empty((1, 2640 + len(actions)))
    for action in actions:
        # csv 파일 경로 유의!!
        for filename in os.listdir("./csv_part/{0}_csv".format(action)):
            print(filename)
            if filename.endswith('.DS_Store'):
                continue
            # csv 파일 경로 유의!!
            file = np.loadtxt(os.path.join(
                "./csv_part/{0}_csv".format(action), filename), delimiter=",", dtype=np.float32)

            file = file.flatten()
            file = np.concatenate([file, makelabel(action)])
            dataset = np.append(dataset, [file], axis=0)

    dataset = np.delete(dataset, (0), axis=0)
    print(actions)
    print(dataset.shape)
    np.savetxt('mydataset_v3.csv', dataset, delimiter=",", fmt='%.5f')


if __name__ == '__main__':
    main()

from configs import *
import sampling20 as smp


def makelabel(action):
    idx = actions.index(action)
    label = np.zeros(len(actions), dtype=np.float32)
    label[idx] = 1.0
    return label


def makeTrainDataset():
    dataset = np.empty(
        (1, config['seq_length'] * config['data_dim'] + len(actions)))
    for action in actions:
        # csv 파일 경로 유의!!
        for filename in os.listdir("./csv_part/{0}_csv".format(action)):
            print(filename)
            if filename.endswith('.DS_Store'):
                continue
            # csv 파일 경로 유의!!
            file = np.loadtxt(os.path.join(
                "./csv_part/{0}_csv".format(action), filename), delimiter=",", dtype=np.float32)

            if config['seq_length'] == 20:
                # 연속된 프레임 샘플링
                starts = [x for x in range(10)]
                for start in starts:
                    sample_file = smp.seqSample20(file, start)
                    sample_file = sample_file.flatten()
                    sample_file = np.concatenate(
                        [sample_file, makelabel(action)])
                    dataset = np.append(dataset, [sample_file], axis=0)

                # 하나의 비디오를 5번 랜덤 샘플링
                for i in range(5):
                    sample_file = smp.randSample20(file)
                    # 샘플링한 20프레임의 데이터를 하나로 펼침
                    sample_file = sample_file.flatten()
                    # 데이터 마지막에 라벨링 추가
                    sample_file = np.concatenate(
                        [sample_file, makelabel(action)])
                    dataset = np.append(dataset, [sample_file], axis=0)
            else:
                # 그냥 30프레임 전부 사용
                file = file.flatten()
                file = np.concatenate([file, makelabel(action)])
                dataset = np.append(dataset, [file], axis=0)

    dataset = np.delete(dataset, (0), axis=0)
    print(actions)
    print(config['seq_length'])
    print(dataset.shape)
    np.savetxt('mydataset_v3_train_{}frames.csv'.format(
        config['seq_length']), dataset, delimiter=",", fmt='%.5f')
    print('Training Dataset for Model V3 Complete!')


def makeValidationDataset():
    dataset = np.empty(
        (1, config['seq_length'] * config['data_dim'] + len(actions)))
    for action in actions:
        # csv 파일 경로 유의!!
        for filename in os.listdir("./csv_part_valid/{0}_csv".format(action)):
            print(filename)
            if filename.endswith('.DS_Store'):
                continue
            # csv 파일 경로 유의!!
            file = np.loadtxt(os.path.join(
                "./csv_part_valid/{0}_csv".format(action), filename), delimiter=",", dtype=np.float32)

            if config['seq_length'] == 20:
                # 연속된 프레임 샘플링
                starts = [x for x in range(10)]
                for start in starts:
                    sample_file = smp.seqSample20(file, start)
                    sample_file = sample_file.flatten()
                    sample_file = np.concatenate(
                        [sample_file, makelabel(action)])
                    dataset = np.append(dataset, [sample_file], axis=0)

                # 하나의 비디오를 5번 랜덤 샘플링
                for i in range(5):
                    sample_file = smp.randSample20(file)
                    # 샘플링한 20프레임의 데이터를 하나로 펼침
                    sample_file = sample_file.flatten()
                    # 데이터 마지막에 라벨링 추가
                    sample_file = np.concatenate(
                        [sample_file, makelabel(action)])
                    dataset = np.append(dataset, [sample_file], axis=0)

            else:
                # 그냥 30프레임 전부 사용
                file = file.flatten()
                file = np.concatenate([file, makelabel(action)])
                dataset = np.append(dataset, [file], axis=0)

    dataset = np.delete(dataset, (0), axis=0)
    print(actions)
    print(config['seq_length'])
    print(dataset.shape)
    np.savetxt('mydataset_v3_valid_{}frames.csv'.format(
        config['seq_length']), dataset, delimiter=",", fmt='%.5f')
    print('Validaton Dataset for Model V3 Complete!')


def makeTestDataset():
    dataset = np.empty(
        (1, config['seq_length'] * config['data_dim'] + len(actions)))
    for action in actions:
        # csv 파일 경로 유의!!
        for filename in os.listdir("./skeleton_csv/csv_modelv3_test/{0}_csv".format(action)):
            print(filename)
            if filename.endswith('.DS_Store'):
                continue
            # csv 파일 경로 유의!!
            file = np.loadtxt(os.path.join(
                "./skeleton_csv/csv_modelv3_test/{0}_csv".format(action), filename), delimiter=",", dtype=np.float32)

            if config['seq_length'] == 20:
                # 연속된 프레임 샘플링
                starts = [x for x in range(10)]
                for start in starts:
                    sample_file = smp.seqSample20(file, start)
                    sample_file = sample_file.flatten()
                    sample_file = np.concatenate(
                        [sample_file, makelabel(action)])
                    dataset = np.append(dataset, [sample_file], axis=0)

                # 하나의 비디오를 5번 랜덤 샘플링
                for i in range(5):
                    sample_file = smp.randSample20(file)
                    # 샘플링한 20프레임의 데이터를 하나로 펼침
                    sample_file = sample_file.flatten()
                    # 데이터 마지막에 라벨링 추가
                    sample_file = np.concatenate(
                        [sample_file, makelabel(action)])
                    dataset = np.append(dataset, [sample_file], axis=0)

            else:
                # 그냥 30프레임 전부 사용
                file = file.flatten()
                file = np.concatenate([file, makelabel(action)])
                dataset = np.append(dataset, [file], axis=0)

    dataset = np.delete(dataset, (0), axis=0)
    print(actions)
    print(config['seq_length'])
    print(dataset.shape)
    np.savetxt('./dataset/mydataset_v3_test_{}frames.csv'.format(
        config['seq_length']), dataset, delimiter=",", fmt='%.5f')
    print('{}frame Test Dataset for Model V3 Complete!'.format(
        config['seq_length']))


def main():
    # makeTrainDataset()
    # makeValidationDataset()
    makeTestDataset()


if __name__ == '__main__':
    main()

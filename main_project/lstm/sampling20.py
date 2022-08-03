from configs import *

# 0~29 중에 무작위로 20개 숫자를 샘플링, 중복 없음


def random20():
    a = np.random.choice(30, 20, replace=False)
    a.sort()
    return a

# start부터 start+19까지 숫자를 반환


def seq20(start):
    res = [x for x in range(30)]
    return res[start:start+20]


# data : 넘파이 배열로 변환한 csv파일(하나의 비디오)
# res : csv파일에서 랜덤으로 샘플링한 20개의 프레임 데이터 (20,88)


def randSample20(data):
    frames = random20()
    res = np.empty((1, config['data_dim']))
    for frame in frames:
        res = np.append(res, [data[frame]], axis=0)
    res = np.delete(res, (0), axis=0)
    res = res.astype(np.float32)
    return res


def seqSample20(data, start):
    frames = seq20(start)
    res = np.empty((1, config['data_dim']))
    for frame in frames:
        res = np.append(res, [data[frame]], axis=0)
    res = np.delete(res, (0), axis=0)
    res = res.astype(np.float32)
    return res

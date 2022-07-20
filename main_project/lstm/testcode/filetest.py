import os

actions = ['squatdown', 'squatup']

cnt = 0

for action in actions:
    for filename in os.listdir("./{0}_csv".format(action)):
        cnt += 1

print(cnt)

# 지금 비디오 데이터셋 몇개 있는지 세주는 코드
# 현재 총 249개 보유

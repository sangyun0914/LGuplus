import os

actions = ['squat-down', 'squat-up', 'pushup-down',
           'pushup-up', 'lunge-down', 'lunge-up']

cnt = 0

for action in actions:
    cnt = 0
    for filename in os.listdir("../csv/{0}_csv".format(action)):
        cnt += 1
    print(action, cnt)

# 지금 비디오 데이터셋 몇개 있는지 세주는 코드
# 현재 총 249개 보유

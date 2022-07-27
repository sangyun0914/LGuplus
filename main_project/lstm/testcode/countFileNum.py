import os

actions = ['squat-down', 'squat-up', 'pushup-down',
           'pushup-up', 'lunge-down', 'lunge-up']

cnt = 0

for action in actions:
    cnt = 0
    for filename in os.listdir("../csv_extra/{0}_csv".format(action)):
        cnt += 1
    print(action, cnt)

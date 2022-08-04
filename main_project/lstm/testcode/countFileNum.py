import os

actions = ['squat-down', 'squat-up', 'pushup-down',
           'pushup-up', 'lunge-down', 'lunge-up', 'stand', 'stand2push', 'push2stand']


print('training dataset')
for action in actions:
    cnt = 0
    for filename in os.listdir("../videos/{}".format(action)):
        if filename.endswith('.DS_Store'):
            continue
        cnt += 1
    print(action, cnt)

print('\nvalidation dataset')
for action in actions:
    cnt = 0
    for filename in os.listdir("../valid_videos/{}".format(action)):
        if filename.endswith('.DS_Store'):
            continue
        cnt += 1
    print(action, cnt)

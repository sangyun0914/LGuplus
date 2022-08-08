import os

actions = ['squat-down', 'squat-up', 'pushup-down',
           'pushup-up', 'lunge-down', 'lunge-up', 'stand', 'stand2push', 'push2stand']

for action in actions:
    path = "./csv_part_valid/{0}_csv".format(action)
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        os.remove(filepath)
        print(filename, 'removed')

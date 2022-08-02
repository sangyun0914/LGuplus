import os

actions = ['squat-down', 'squat-up', 'pushup-down',
           'pushup-up', 'lunge-down', 'lunge-up']

for action in actions:
    path = "../csv_part/{0}_csv".format(action)
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        os.remove(filepath)
        print(filename, 'removed')

import os

for filename in os.listdir("squatdown"):
    with open(os.path.join("squat", filename), 'r') as f:
        text = f.read()
        print(text)

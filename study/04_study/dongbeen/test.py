import pickle

with open("train_label.pkl", "rb") as fr:
    data = pickle.load(fr)

print(data)
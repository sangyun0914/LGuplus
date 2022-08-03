import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import mediapipe as mp
import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt
import timeit

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


actions = ['squat-down', 'squat-up', 'pushup-down',
           'pushup-up', 'lunge-down', 'lunge-up', 'stand']

config = {}

config['seq_length'] = 30  # 20 프레임
config['data_dim'] = 88
config['hidden_dim'] = 20
config['output_dim'] = len(actions)  # 운동 종류
config['lstm_layers'] = 1
config['dropout'] = 0

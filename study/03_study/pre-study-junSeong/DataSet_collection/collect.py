from sqlite3 import DatabaseError
from datetime import datetime
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import uuid

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Data') 

# Actions that we try to detect
actions = np.array(['lunge-down','lunge-up','crunch-down','crunch-up'])

# Thirty videos worth of data
no_sequences = 3

# Videos are going to be 30 frames in length
sequence_length = 20

cap = cv2.VideoCapture(0)
# Set mediapipe model 

# NEW LOOP
# Loop through actions
action_num = 1
for action in actions:
    # Loop through sequences aka videos
    for sequence in range(no_sequences):
        # Loop through video length aka sequence length
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')

        # Read feed
        width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = int(fps)
        #                                       운동번호 프레임 날짜 번호
        videopath = os.path.join(DATA_PATH,action,'{}-{}-{}-{}.avi'.format(str('0'+str(action_num)),str(fps),datetime.today().strftime('%Y-%m-%d'),str(sequence)))
        print('save to ', videopath)
        out = cv2.VideoWriter(videopath, fourcc, fps, (width,height))

        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            # NEW Apply wait logic
            if frame_num == 0: 
                print('STARTING COLLECTION for {} wait...'.format(action))
                cv2.waitKey(2000)
                print('Collecting frames for {} Video Number {}'.format(action, sequence))
                # Show to screen
                cv2.imshow('Data collection', frame)
                out.write(frame) # 영상데이터만 저장 (소리 X)

            else: 
                print('Collecting frames for {} Video Number {}'.format(action, sequence))
                # Show to screen
                cv2.imshow('OpenCV Feed', frame)
                out.write(frame) # 영상데이터만 저장 (소리 X)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


    action_num += 1
cap.release()
cv2.destroyAllWindows()

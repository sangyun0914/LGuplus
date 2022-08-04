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
actions = np.array(['lunge-down', 'lunge-up'])

# Thirty videos worth of data
no_sequences = 12

# Videos are going to be 30 frames in length
sequence_length = 30

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 0)
cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 0)
# Set mediapipe model

# NEW LOOP
# Loop through actions
action_num = 5
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

        #                                       운동번호 일련번호 번호 프레임 (true or false)
        videopath = os.path.join(DATA_PATH, action, '{}-{}-{}-{}-{}.avi'.format(str('0'+str(
            action_num)), datetime.today().strftime('%Y%m%d%H%M'), str(sequence), str(fps), str(0)))
        videopath_flip = os.path.join(DATA_PATH, action, '{}-{}-{}-{}-{}.avi'.format(str('0'+str(
            action_num)), datetime.today().strftime('%Y%m%d%H%M'), str(sequence), str(fps), str(1)))

        out = cv2.VideoWriter(videopath, fourcc, fps, (width, height))
        out_flip = cv2.VideoWriter(
            videopath_flip, fourcc, fps, (width, height))

        for frame_num in range(sequence_length):
            if (frame_num == 0):
                print('STARTING COLLECTION for {} (video number {}) wait...'.format(
                    action, sequence))
                cv2.waitKey(2000)
            ret, frame = cap.read() #이 맥북의 정보는 내가 가져가겠다 뭐지 아 ???
            show_frame = frame.copy()

            print('Collecting frames for {} Video Number {}'.format(action, sequence))
            cv2.putText(show_frame, 'Collecting frames for {} Video Number {}'.format(
                action, sequence), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.imshow('Realtime view', show_frame)

            frame_flip = cv2.flip(frame, 1)
            # NEW Apply wait logic

            if (frame_num == 0):
                out.write(frame)  # 영상데이터만 저장 (소리 X)
                out_flip.write(frame_flip)
            else:
                out.write(frame)  # 영상데이터만 저장 (소리 X)
                out_flip.write(frame_flip)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        print('save to ', videopath)
        print('save to ', videopath_flip)

        cv2.waitKey(1000)
    action_num += 1

out.release()
out_flip.release()
cap.release()
cv2.destroyAllWindows()

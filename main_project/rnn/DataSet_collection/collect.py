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
no_sequences = 5

# Videos are going to be 30 frames in length
sequence_length = 30

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
        fps_low = int(fps * 0.7)
        fps_high = int(fps * 1.3)

        #                                       운동번호 일련번호 번호 프레임
        videopath = os.path.join(DATA_PATH,action,'{}-{}-{}-{}.avi'.format(str('0'+str(action_num)),datetime.today().strftime('%Y%m%d%H%M'),str(sequence),str(fps)))
        videopath_low = os.path.join(DATA_PATH,action,'{}-{}-{}-{}.avi'.format(str('0'+str(action_num)),datetime.today().strftime('%Y%m%d%H%M'),str(sequence),str(fps_low)))
        videopath_high = os.path.join(DATA_PATH,action,'{}-{}-{}-{}.avi'.format(str('0'+str(action_num)),datetime.today().strftime('%Y%m%d%H%M'),str(sequence),str(fps_high)))

        out = cv2.VideoWriter(videopath, fourcc, fps, (width,height))
        out_low = cv2.VideoWriter(videopath_low, fourcc, fps_low, (width,height))
        out_high = cv2.VideoWriter(videopath_high, fourcc, fps_high, (width,height))

        print('Collecting frames for {} Video Number {}'.format(action, sequence))
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            show_frame = frame.copy()

            print('Collecting frames for {} Video Number {}'.format(action, sequence))
            cv2.putText(show_frame,'Collecting frames for {} Video Number {}'.format(action, sequence), (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 1)
            cv2.imshow('Realtime view',show_frame)
            # NEW Apply wait logic
            if frame_num == 0: 
                print('STARTING COLLECTION for {} wait...'.format(action))
                # cv2.imshow('Data collection', frame)
                cv2.waitKey(5000)
                # Show to screen

                out.write(frame) # 영상데이터만 저장 (소리 X)
                out_low.write(frame)
                out_high.write(frame)

            else: 
                # Show to screen
                # cv2.imshow('Data collection', frame)
                out.write(frame) # 영상데이터만 저장 (소리 X)
                out_low.write(frame)
                out_high.write(frame)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        print('save to ', videopath)
        print('save to ', videopath_low)
        print('save to ', videopath_high)
        cv2.waitKey(2000)
    action_num += 1

out.release()
out_high.release()
out_low.release()
cap.release()
cv2.destroyAllWindows()

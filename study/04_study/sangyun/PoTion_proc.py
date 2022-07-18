import cv2
import numpy as np
import mediapipe as mp
import pathlib


DIR = str(pathlib.Path(__file__).parent.resolve())


cap = cv2.VideoCapture(DIR + "/video/squat.mp4")

nFrames = 0
FRAME_SIZE_X= 0
FRAME_SIZE_Y=0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break
    #FRAME_SIZE_X, FRAME_SIZE_Y , _ = image.size()
        



# get heatmap
#heatmaps=np.zeros((nFrames,NUM_JOINT,FRAME_SIZE,FRAME_SIZE))
heatmaps=np.zeros((30,33,150,150))
print(heatmaps)

ind=0
for i in range(nFrames):
    for j in range(NUM_JOINT):
        g2d = Gaussian2D(amplitude=rand_amplitudes[ind],x_mean=normalized_y[i,j], y_mean=normalized_x[i,j], 
                x_stddev=rand_stddev_x[ind], y_stddev=rand_stddev_y[ind]) 
        ind+=1
        heatmaps[i,j,:,:]=g2d(*np.mgrid[0:FRAME_SIZE, 0:FRAME_SIZE])
        
# get descriptor
each_label=each_label.split('.')[0]
descriptor=get_descriptor(heatmaps,C).astype('float32')
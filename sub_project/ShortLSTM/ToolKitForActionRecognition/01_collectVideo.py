import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten() if results.pose_world_landmarks else np.zeros(33*4)

    return pose

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
action = "pushup"
VIDEO_PATH = 'fullVideo/fullpushup.mp4'

cap = cv2.VideoCapture(VIDEO_PATH)
no_sequences = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    else:
        no_sequences += 1
        print("COUNTING FRAME NUM...{}".format(no_sequences))

cap.release()
cv2.destroyAllWindows()
# Thirty videos worth of data

sequence_length = 30

for sequence in range(no_sequences):
    try: 
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
    except:
        pass

cap = cv2.VideoCapture(VIDEO_PATH)
# Set mediapipe model 
with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5) as pose:
    # Loop through sequences aka videos
    for sequence in range(no_sequences):
        # Loop through video length aka sequence length
        for frame_num in range(sequence_length):

            # Read feed
            ret, frame = cap.read()
        
            if not ret:
                break

            # Make detections
            image, results = mediapipe_detection(frame, pose)
#                 print(results)

            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # NEW Export keypoints
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            cv2.imshow("data collect",image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()

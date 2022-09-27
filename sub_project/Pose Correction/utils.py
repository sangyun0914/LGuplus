import cv2
import mediapipe as mp
import numpy as np
from constants import PoseLandmark

mp_pose = mp.solutions.pose


def extract_landmarks(video_path):
    # For webcam input:
    cap = cv2.VideoCapture(video_path)
    total_landmarks = []
    ind =0
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            ind += 1
            landmarks = []
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            for landmark in PoseLandmark:
                joint = results.pose_world_landmarks.landmark._values[landmark.value]
                landmarks.append(joint)

            total_landmarks.append(landmarks)

        poses = np.array(total_landmarks)
        print("frame: ", ind)
        cap.release()
        return poses

def inference_landmarks(image):
    # For static images:
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        #image = cv2.imread(frame)
        landmarks = []

        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return None

        for landmark in PoseLandmark:
            joint = results.pose_world_landmarks.landmark._values[landmark.value]
            landmarks.append(joint)

        poses = np.array(landmarks)
        return poses      

#extract_landmarks("sub_project/Pose Correction/video/squat.mp4")

def dist(joint1, joint2):
    return np.sqrt(np.square(joint1.x - joint2.x) + np.square(joint1.y - joint2.y))


def load_ps(filename):
    pass
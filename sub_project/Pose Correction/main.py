"""Pose Correction main script."""

import argparse
import os
import numpy as np
import cv2

from evaluate import evaluate_angle
import utils
from constants import PoseLandmark
import constants

def main():
    parser = argparse.ArgumentParser(description='Pose Correction')
    parser.add_argument('--mode', type=str, default='evaluate', help='Pose Trainer application mode.\n'
            'One of evaluate, batch_json, evaluate_npy. See the code for more info.')
    parser.add_argument('--input_folder', type=str, default='videos', help='(Used by the batch_json mode only)\n'
            'Input folder for videos.\n'
            'Defaults to the videos folder in this repository folder.')
    parser.add_argument('--output_folder', type=str, default='poses', help='(Used by the batch_json mode only)\n'
            'Folder for pose JSON files.\n'
            'Defaults to the poses folder in this repository folder.')
    parser.add_argument('--video', type=str, help='(Used by the evaluate mode only)\n'
            'Input video filepath for evaluation. Looks for it in the root folder of the repository.')
    parser.add_argument('--file', type=str, help='(Used by the evaluate_npy mode only)\n'
            'Full path to the input .npy file for evaluation.')
    parser.add_argument('--exercise', type=str, default='squat', help='Exercise type to evaluate.')

    args = parser.parse_args()

    if args.mode == 'evaluate':
        if args.video:
            print('processing video file...')
            video = os.path.basename(args.video)
            
            # Run OpenPose on the video, and write a folder of JSON pose keypoints to a folder in
            # the repository root folder with the same name as the input video.
            output_path = os.path.join('..', os.path.splitext(video)[0])

            pose_seq = utils.extract_landmarks(args.video)

            (correct, feedback) = evaluate_angle(pose_seq, args.exercise)
            if correct:
                print('Exercise performed correctly!')
            else:
                print('Exercise could be improved:')
            print(feedback)
        else:
            print('No video file specified.')
            return
  

    # Evaluate the .npy file as a pose sequence for the specified exercise.
    elif args.mode == 'evaluate_npy':
        if args.file:
            pose_seq = utils.load_ps(args.file)
            (correct, feedback) = evaluate_angle(pose_seq, args.exercise)
            if correct:
                print('Exercise performed correctly:')
            else:
                print('Exercise could be improved:')
            print(feedback)
        else:
            print('No npy file specified.')
            return
    
    # Evaluate on real-stream cam
    elif args.mode == 'inference':
        #cap = cv2.VideoCapture(0)

        ind = 0
        cap = cv2.VideoCapture('sub_project/Pose Correction/video/squat.mp4')
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            cv2.imshow("image", image)
            cv2.waitKey(5)
            pose_seq = utils.inference_landmarks(image)
            
            if pose_seq is None:
                continue

            left_joints = [pose_seq[PoseLandmark.LEFT_SHOULDER], pose_seq[PoseLandmark.LEFT_HIP], pose_seq[PoseLandmark.LEFT_KNEE], pose_seq[PoseLandmark.LEFT_ANKLE]]
            
            # filter out data points where a part does not exist
            if not all(joint.visibility > constants.VIS_THRESHOLD for joint in left_joints):
                continue
            
            ind += 1
            left_torso = np.array([left_joints[0].x - left_joints[1].x, left_joints[0].y - left_joints[1].y, left_joints[0].z - left_joints[1].z])
            left_vertical = np.array([left_joints[1].x, left_joints[1].y, left_joints[1].z - 1])

            # normalize vectors
            left_torso = left_torso / np.expand_dims(np.linalg.norm(left_torso, axis=1), axis=1)
            left_vertical = left_vertical / np.expand_dims(np.linalg.norm(left_vertical, axis=1), axis=1)
            #torso_vecs = torso_vecs / np.expand_dims(np.linalg.norm(torso_vecs, axis=1), axis=1)
            #forearm_vecs = forearm_vecs / np.expand_dims(np.linalg.norm(forearm_vecs, axis=1), axis=1)

            left_back_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(left_torso, left_vertical), axis=1), -1.0, 1.0)))
            #upper_arm_forearm_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(upper_arm_vecs, forearm_vecs), axis=1), -1.0, 1.0)))

            # use thresholds learned from analysis
            #upper_arm_torso_range = np.max(upper_arm_torso_angles) - np.min(upper_arm_torso_angles)
            #upper_arm_forearm_min = np.min(upper_arm_forearm_angles)

            #print('Upper arm and torso angle range: {}'.format(upper_arm_torso_range))
            #print('Upper arm and forearm minimum angle: {}'.format(upper_arm_forearm_min))
            print(left_back_angles)
            #print()
            #print('Max: {}'.format(np.max(left_back_angles)))
            #print('Min: {}'.format(np.min(left_back_angles)))
        
        print(ind)
        cap.release()


    
    else:
        print('Unrecognized mode option.')
        return




if __name__ == "__main__":
    main()


'''
run: python3 main.py --mode evaluate --video video/squat.mp4            

'''
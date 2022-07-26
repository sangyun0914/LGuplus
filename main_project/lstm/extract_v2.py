# extract.py에서 추가적으로 피쳐를 넣어서 저장하는 버전

import cv2
import numpy as np
import mediapipe as mp
import copy
import os
import extraFeatures

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 운동 종류 지정
"""actions = ['squat-down', 'squat-up', 'pushup-down',
           'pushup-up', 'lunge-down', 'lunge-up']"""
actions = ['pushup-down', 'pushup-up', 'lunge-down', 'lunge-up']

# video_path : 스켈레톤을 추출하려고 하는 비디오의 경로
# video_name : 비디오의 파일 이름, csv파일 이름을 비디오 이름과 통일해주기 위해서 필요함
# csv_path : 추출된 csv 파일이 저장될 경로


def extractPose(video_path, video_name, csv_path):
    # 프레임마다 뽑힌 스켈레톤 좌표를 하나로 모으기 위하여 비어있는 넘파이 배열 생성
    # 22개 랜드마크의 좌표, visibility 총 88개
    # 관절 사이 각도 총 8개
    extract = np.empty((1, 96))

    # 비디오 캡쳐 시작
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Action : {0} / filename : {1} / Skeleton Extraction Finished".format(
                    csv_path[12:-4], video_name))
                break

            # 미디어파이프를 이용하여 스켈레톤 추출
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            temp = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten(
            ) if results.pose_world_landmarks else np.zeros(132)

            # print(video_name)
            angles = extraFeatures.extractAngles(results)

            # 얼굴을 제외한 22개의 랜드마크만 사용하기 위해 temp의 0~43번 인덱스 내용은 버림
            # temp뒤에 8개의 관절 사이 각도 추가
            temp2 = np.append(temp[44:], angles)
            extract = np.append(extract, [temp2], axis=0)

    # 첫번째 열은 아무 의미 없는 값이 들어가있기 때문에 지워줌
    extract = np.delete(extract, (0), axis=0)
    extract = extract.astype(np.float32)
    # print(extract.shape)

    # 30 프레임에서 추출한 관절 정보들을 하나의 csv 파일로 저장
    # 소수 다섯번째 자리까지만 저장
    out_path = os.path.join(csv_path, video_name) + '.csv'
    np.savetxt(out_path, extract, delimiter=",", fmt='%.5f')
    cap.release()


def main():
    # 지정한 운동들을 스켈레톤 추출하여 저장
    for action in actions:
        # 원본 영상
        for filename in os.listdir("./videos/{0}".format(action)):
            if filename.endswith('.DS_Store'):
                continue
            video_path = os.path.join("./videos/{0}".format(action), filename)
            csv_path = os.path.join('./csv_extra/{0}_csv'.format(action))
            extractPose(video_path, filename, csv_path)

        # 90도 회전 영상
        for filename in os.listdir("./videos_rotate90/{0}".format(action)):
            if filename.endswith('.DS_Store'):
                continue
            video_path = os.path.join(
                "./videos_rotate90/{0}".format(action), filename)
            csv_path = os.path.join('./csv_extra/{0}_csv'.format(action))
            extractPose(video_path, filename+'90', csv_path)

        # 270도 회전 영상
        for filename in os.listdir("./videos_rotate270/{0}".format(action)):
            if filename.endswith('.DS_Store'):
                continue
            video_path = os.path.join(
                "./videos_rotate270/{0}".format(action), filename)
            csv_path = os.path.join('./csv_extra/{0}_csv'.format(action))
            extractPose(video_path, filename+'270', csv_path)


if __name__ == '__main__':
    main()

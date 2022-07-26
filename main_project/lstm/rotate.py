# 동영상 90도 돌려서 저장

import cv2
import numpy as np
import os

actions = ['squat-down', 'squat-up', 'pushup-down',
           'pushup-up', 'lunge-down', 'lunge-up']


def rotateVideo(video_path, video_name, out_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(out_path, fourcc, fps, (height, width))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # print('end of video')
            break
        rotate = frame.copy()
        rotate = cv2.rotate(rotate, cv2.ROTATE_90_CLOCKWISE)
        out.write(rotate)
        if cv2.waitKey(1) > 0:
            break

    out.release()
    cap.release()
    print(video_name, 'rotation complete')


def main():
    for action in actions:
        for filename in os.listdir("./videos/{0}".format(action)):
            video_path = os.path.join("./videos/{0}".format(action), filename)
            out_path = os.path.join(
                "./videos_rotate90/{0}".format(action), filename)
            rotateVideo(video_path, filename, out_path)


if __name__ == '__main__':
    main()

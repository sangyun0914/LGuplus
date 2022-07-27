# 동영상 90도 혹은 270도 돌려서 저장

import cv2
import numpy as np
import os

actions = ['squat-down', 'squat-up', 'pushup-down',
           'pushup-up', 'lunge-down', 'lunge-up']


def rotateVideo(video_path, video_name, out_path, degree):
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
        if degree == 90:
            rotate = cv2.rotate(rotate, cv2.ROTATE_90_CLOCKWISE)
        elif degree == 270:
            rotate = cv2.rotate(rotate, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out.write(rotate)
        if cv2.waitKey(1) > 0:
            break

    out.release()
    cap.release()
    print(video_name, 'rotation complete')


def main():
    for action in actions:
        for filename in os.listdir("./videos/{0}".format(action)):
            if filename.endswith('.DS_Store'):
                continue
            video_path = os.path.join("./videos/{0}".format(action), filename)
            out_path1 = os.path.join(
                "./videos_rotate90/{0}".format(action), filename)
            out_path2 = os.path.join(
                "./videos_rotate270/{0}".format(action), filename)
            rotateVideo(video_path, filename, out_path1, 90)
            rotateVideo(video_path, filename, out_path2, 270)


if __name__ == '__main__':
    main()

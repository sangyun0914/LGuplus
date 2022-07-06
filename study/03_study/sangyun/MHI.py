"""
Example to extract motion history images using opencv2. stripped from opencv2 python examples motempl.py
link to the gif: https://giphy.com/gifs/bJDYIRToRpkEU
command to extract the jpgs: convert example.gif -coalesce images/example-%03d.jpg
You have to use fixed length pattern for image sequence, such as ./images/example-%03d.jpg
"""

import numpy as np
import cv2

MHI_DURATION = 50
DEFAULT_THRESHOLD = 32


def main():

    video_src = 0
    cv2.namedWindow('motion-history')
    cv2.namedWindow('raw')
    cv2.moveWindow('raw', 200, 0)

    cam = cv2.VideoCapture(video_src)
    ret, frame = cam.read()
    h, w = frame.shape[:2]
    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)
    timestamp = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame_diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        ret, fgmask = cv2.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
        timestamp += 1
        
        # update motion history
        cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)
        # cv.motempl.updateMotionHistory(silhouette, mhi, timestamp, duration) ->	mhi
            # Parameters:
            # silhouette	Silhouette mask that has non-zero pixels where the motion occurs.
            # mhi	Motion history image that is updated by the function (single-channel, 32-bit floating-point).
            # timestamp	Current time in milliseconds or other units.
            # duration	Maximal duration of the motion track in the same units as timestamp .


        # normalize motion history
        mh = np.uint8(np.clip((motion_history - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)
        cv2.imshow('motion-history', mh)
        cv2.imshow('raw', frame)

        prev_frame = frame.copy()
        if 0xFF & cv2.waitKey(5) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
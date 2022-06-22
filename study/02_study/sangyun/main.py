import torch
import numpy as np
import cv2
from time import time

class detection:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # or yolov5n - yolov5x6, custom
        self.classes = self.model.names
        self.device = 'cpu'

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def plot_boxes(self, labels, cord, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        n = len(labels)
        count = 0
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            if labels[i] == 0:
                count += 1
                row = cord[i]
                if row[4] >= 0.3:
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    bgr = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                    cv2.putText(frame, self.classes[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame, count

    def __call__(self):
        cap = cv2.VideoCapture('video.mp4')
        ret, frame = cap.read()
        assert ret

        x,y,w,h	= cv2.selectROI('img', frame, False)
        if w and h:
            roi = frame[y:y+h, x:x+w]

        cv2.destroyWindow("img")

        assert cap.isOpened()
    
        while True:
        
            ret, frame = cap.read()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            assert ret
            roi = frame[y:y+h, x:x+w]
           
            start_time = time()
            labels, cord = self.score_frame(roi)
            # print("results:", (results))
            frame[y:y+h, x:x+w], count = self.plot_boxes(labels, cord, roi)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
            print(f"Frames Per Second : {fps}")
            print("count: ", count)
            if (count != 0):
                cv2.putText(frame, f'Detected!', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            
            cv2.imshow('YOLOv5 Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cap.release()

detector = detection()
detector()

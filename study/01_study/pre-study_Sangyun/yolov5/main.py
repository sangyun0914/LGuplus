import os
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (LOGGER, check_img_size, cv2, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync

# Model
weights=ROOT / 'yolov5s.mlmodel'  # model.pt path(s)
source= 0  # 0 for webcam
imgsz=(640, 640)  # inference size (height, width)
max_det=1000  # maximum detections per image
device=torch.device('cpu')  # cuda device, i.e. 0 or 0,1,2,3 or cpu
view_img=True  # show results
save_crop=False  # save cropped prediction boxes
nosave=True  # do not save images/videos
classes=None  # filter by class: --class 0, or --class 0 2 3
augment=False  # augmented inference
visualize=False  # visualize features
update=False  # update all models
project=ROOT / 'runs/detect'  # save results to project/name
name='exp'  # save results to project/name
line_thickness=3  # bounding box thickness (pixels)
hide_labels=False  # hide labels
hide_conf=False  # hide confidences
half=False  # use FP16 half-precision inference
dnn=False  # use OpenCV DNN for ONNX inference
coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

source = '0'
save_img = not nosave and not source.endswith('.txt')
save_dir = ROOT / 'runs/detect/exp'


# Load model
model = DetectMultiBackend(weights = weights, device=device)
names = coco_names
stride, pt = model.stride, model.pt
imgsz = check_img_size([640,640], s=stride)  # check image size

dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
bs = len(dataset)  # batch_size


# Run inference
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
seen = 0
for path, im, im0s, vid_cap, s in dataset:
        original = im.copy()
        t1 = time_sync()
        im = torch.from_numpy(im).to(torch.device('cpu'))
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()

        # Inference
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()

        # NMS
        pred = non_max_suppression(pred, conf_thres = 0.25, iou_thres = 0.45, classes = None, agnostic = False, max_det=300)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            # cv2.imshow('original', original)
            cv2.imshow('yolov5s.mlmodel', im0)
            cv2.waitKey(1)  # 1 millisecond


        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        fpsm = 1 / (t3 - t2)
        LOGGER.info(f'FPS: {fpsm:.1f}')

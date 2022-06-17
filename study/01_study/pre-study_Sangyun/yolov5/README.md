## <div align="center">Installation</div>
```bash
pip install -r requirements.txt # install pre-requisite modules
```


## <div align="center">Commands</div>

### Yolov5 using webcam

#### Simplified model: 
```python
python main.py
```
 
#### Full model:
```python
python detect.py --weights yolov5s.mlmodel --source 0 
```

### Yolov5 Inference
```python
python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
                          path/*.jpg  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

# Created environment for running yolov4 with tensorflow
# Enables GPU acceleration for silicon Apple chip
# Modify conda-gpu.yml for other GPU model usage

# Creating environment
conda env create -f conda-gpu.yml

conda activate yolov4-gpu

conda install -c apple tensorflow-deps     
conda install -c apple tensorflow-deps==2.6.0    
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal

conda install easydict -c conda-forge 
python convert_trt.py --weights ./checkpoints/yolov4.tf --quantize_mode float16 --output ./checkpoints/yolov4-trt-fp16-416


# Running cam detection model
python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi


# Result: Does not perform as well in terms of FPS 
# Average 7.0 FPS in M1 Pro Metal
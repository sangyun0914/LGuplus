# 1. Created environment for running yolov4 with tensorflow
Enables GPU acceleration for silicon Apple chip
Modify conda-gpu.yml for other GPU model usage

# Creating environment
conda env create -f conda-gpu.yml

conda activate yolov4-gpu

conda install -c apple tensorflow-deps     
conda install -c apple tensorflow-deps==2.6.0    
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal

conda install easydict -c conda-forge 
python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --input_size 416 --model yolov4 --tiny


# Running cam detection model
python detect_video.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi

# Result: Performs well in terms of FPS, but fluctuating detections, low accuracy
Average 100 FPS in M1 Pro Metal
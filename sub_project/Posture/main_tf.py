import utils.Engine as Engine
import cv2 # Import opencv

if __name__ == "__main__":
  MODEL = "model/ActionNV1.h5"
  VIDEO_PATH = "RandomVideo/DB_SSPPPPLLLS.mov"

  # TESTCAM
  cap = cv2.VideoCapture(VIDEO_PATH)

  # REALTIME CAM
  # cap = cv2.VideoCapture(0)

  #Engine.StartEngine(cap)
  Engine.TF_InferenceEngine(cap,MODEL)
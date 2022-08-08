import utils.Engine as Engine
import cv2 # Import opencv

if __name__ == "__main__":
  MODEL = "model/ActionV7_gb.pkl"
  VIDEO_PATH = "RandomVideo/SY_LLLSSSSLPP.mp4"

  # TESTCAM
  # cap = cv2.VideoCapture(VIDEO_PATH)

  # REALTIME CAM
  cap = cv2.VideoCapture(0)

  # Engine.StartEngine(cap)
  Engine.InferenceEngine(cap,MODEL)
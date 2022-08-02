import utils.Engine as Engine
import cv2 # Import opencv

if __name__ == "__main__":
  MODEL = "model/ActionV6_rf.pkl"

  # TESTCAM
  cap = cv2.VideoCapture("video/pushup2.mp4")

  # REALTIME CAM
  # cap = cv2.VideoCapture(0)

  # Engine.StartEngine(cap)
  Engine.InferenceEngine(cap,MODEL)
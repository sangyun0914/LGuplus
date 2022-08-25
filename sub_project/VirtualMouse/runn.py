import cv2
from utils import Run

if __name__ == "__main__":
  cap = cv2.VideoCapture(0)
  # Music | Record | Return | Feedback | challenge
  function_list = ['Music','Record','Return','Feedback','Challenge']

  Run.main(cap,function_list)
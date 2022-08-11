import utils.Engine as Engine
import utils.Test as Test
import utils.RandomTest as RT
import cv2 # Import opencv
import os

if __name__ == "__main__":
  MODEL = "model/ActionNV2.h5"
  cur_path = os.getcwd()
  test_path = Test.ReturnTestPath("RandomVideo")
  print(test_path)
  test_list = Test.ReturnTestList(test_path)
  RT.RunRandomTest(test_list,MODEL)
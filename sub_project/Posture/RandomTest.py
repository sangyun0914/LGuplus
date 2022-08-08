import utils.Engine as Engine
import utils.Test as Test
import utils.RandomTest as RT
import cv2 # Import opencv
import os

if __name__ == "__main__":
  MODEL = "model/ActionV7_gb.pkl"
  cur_path = os.getcwd()
  folder_name = "RandomBackup"
  test_path = Test.ReturnTestPath(folder_name)
  print(test_path)
  test_list = Test.ReturnTestList(test_path)
  RT.RunRandomTest(test_list,MODEL,folder_name)
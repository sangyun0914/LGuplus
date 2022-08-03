import utils.Engine as Engine
import utils.Test as Test
import cv2 # Import opencv
import os

if __name__ == "__main__":
  cur_path = os.getcwd()
  test_path = Test.ReturnTestPath("TestVideo")
  test_list = Test.ReturnTestList(test_path)

  correct = 0
  for i in test_list:
    MODEL = "model/ActionV6_rf.pkl"
    print("===========================================================")
    VIDEO_PATH = os.path.join("TestVideo",i)
    print("CURRENT VIDEO PATH : {}".format(VIDEO_PATH))
    print("===========================================================")
    cap = cv2.VideoCapture(VIDEO_PATH)
    predict = Engine.TestEngine(cap,MODEL)
    print("Answer  is {}".format(Test.EvalAnswer(i)))
    print("Predict is {}".format(predict))

    if predict == Test.EvalAnswer(i):
      correct+=1
  
  print("===========================================================")
  print("Test Accuracy -> {}%".format(correct/len(test_list)*100))


  # TESTCAM
  # cap = cv2.VideoCapture(VIDEO_PATH)

  # REALTIME CAM
  # cap = cv2.VideoCapture(0)
  
  # Engine.StartEngine(cap)
  # Engine.InferenceEngine(cap,MODEL)
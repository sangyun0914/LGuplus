import utils.Engine as Engine
import utils.Test as Test
import cv2 # Import opencv
import os

def EvalAnswer(name):
  if ("SLP") in name:
    return  "SSLLPP"
  elif ("SPL") in name:
    return "SSPPLL"
  elif ("PSL") in name:
    return "PPSSLL"
  elif ("PLS") in name:
    return "PPLLSS"
  elif ("LSP") in name:
    return "LLSSPP"
  elif ("LPS") in name:
    return "LLPPSS"

if __name__ == "__main__":
  cur_path = os.getcwd()
  test_path = Test.ReturnTestPath("TestVideo")
  test_list = Test.ReturnTestList(test_path)
  
  notcorrect = []
  correct = 0
  for i in test_list:
    print("===========================================================")
    VIDEO_PATH = os.path.join("TestVideo",i)
    print("CURRENT VIDEO PATH : {}".format(VIDEO_PATH))
    print("===========================================================")
    cap = cv2.VideoCapture(VIDEO_PATH)
    predict = Engine.TestCNNEngine(cap)
    print("Answer  is {}".format(EvalAnswer(i)))
    print("Predict is {}".format(predict))

    if predict == EvalAnswer(i):
      correct+=1
    else:
      notcorrect.append((i,EvalAnswer(i),predict))
  
  print("===========================================================")
  print("### TEST END ###")
  print("# Test Accuracy -> {}%".format(correct/len(test_list)*100))
  print("# Total -> {}/{}".format(correct,len(test_list)))
  print("### ANALYSIS ###")
  for index,i in enumerate(notcorrect):
    print("#{} CASE".format(index+1))
    print(" -> Videopath : {}".format(i[0]))
    print(" -> Answer : {}".format(i[1]))
    print(" -> Predict : {}".format(i[2]))
  print("===========================================================")
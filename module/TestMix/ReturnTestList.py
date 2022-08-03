import os

from requests import delete

# ReturnTestPath를 통해 얻게된 TestPath를 input으로 넣으면 됨
def ReturnTestList(test_path):
  test_list = os.listdir(test_path)

  for index, file in enumerate(test_list):
    if ".DS_Store" in file:
      test_list.pop(index)

  return test_list
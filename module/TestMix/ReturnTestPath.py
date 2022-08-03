import os

# input으로 TestVideo가 있는 폴더 이름을 넣으면 됨 (상대좌표로)
def ReturnTestPath(TestVideo):
  cur_path = os.getcwd()
  test_path = os.path.join(cur_path,TestVideo)

  return test_path
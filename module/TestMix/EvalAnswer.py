# TestVideo 파일이름을 input으로 넣으면 올바른 정답 return
def EvalAnswer(name):
  if ("SLP") in name:
    return "SSLLPP"
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
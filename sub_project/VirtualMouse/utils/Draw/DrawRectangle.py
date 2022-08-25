import cv2

def DrawRectangle(image,COLOR,THICKNESS,opt_list,block):
  for i in range(block):
    index = 4*i
    cv2.rectangle(image, (opt_list[index],opt_list[index+1]), (opt_list[index+2],opt_list[index+3]), COLOR, THICKNESS)
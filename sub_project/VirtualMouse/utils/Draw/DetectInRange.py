import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import analysis as A

def DetectInRange(image,COLOR,point_x,point_y,opt_list,block):
  for i in range(block):
    index = 4*i 
    if (A.IsBetween(point_x,point_y,opt_list[index],opt_list[index+1],opt_list[index+2],opt_list[index+3])):
      cv2.rectangle(image, (opt_list[index],opt_list[index+1]), (opt_list[index+2],opt_list[index+3]), COLOR, cv2.FILLED)
  return image
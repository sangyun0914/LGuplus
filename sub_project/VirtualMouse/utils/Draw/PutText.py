import cv2

def PutText(image,opt_list,function_list,GAP,COLOR):
  for i in range(len(function_list)):
    index = 4*i
    cv2.putText(image, function_list[i] , (opt_list[index]+GAP,opt_list[index+1]+GAP), cv2.FONT_HERSHEY_SIMPLEX, 1,COLOR, 2, cv2.LINE_AA)
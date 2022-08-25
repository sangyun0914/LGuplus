import cv2

def ExtractFinger(image,results,RADIUS,COLOR):
  point = results.multi_hand_landmarks[0].landmark[8]
  height, width = image.shape[0], image.shape[1]
  point_x,point_y = int(point.x * width),int(point.y * height)

  COLOR = (0,255,0)

  cv2.circle(image, (point_x,point_y), RADIUS , COLOR , cv2.FILLED , cv2.LINE_AA) #속이 꽉 찬 

  return image,point_x,point_y
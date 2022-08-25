def FindCoordList(image,opt_list,block,height_ratio,width_ratio):
  x, y = 0, 0
  # start_x, start_y, end_x, end_y
  for i in range(block):
    opt_list.append(x)
    opt_list.append(y)
    x += width_ratio
    opt_list.append(x)
    opt_list.append(height_ratio)
  return opt_list
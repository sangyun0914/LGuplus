def IsBetween(target_x,target_y,start_x,start_y,end_x,end_y):
  Between_x = False
  Between_y = False

  if (start_x <= target_x<= end_x):
    Between_x = True
  else:
    Between_x = False

  if (start_y <= target_y<= end_y):
    Between_y = True
  else:
    Between_y = False
  
  if (Between_x and Between_y):
    return True
  else:
    return False
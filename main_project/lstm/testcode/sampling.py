import numpy as np

# 30프레임 중에서 무작위로 15프레임을 샘플링
a = np.random.choice(30, 15, replace=False)
a.sort()
print(a)

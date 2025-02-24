import numpy as np

# zeres(), ones(): 주어진 모양(shape)에 대해 모든 요소가 0 또는 1일 배열을 생성하는 함수
# 3행 4열
ones = np.ones((3,4))
print(ones)
print()

zeros = np.zeros((3,4))
print(zeros)
# zeros = np.zeros((3,4))
zeros = np.zeros((3,4), dtype = np.int16)

# ndim
# dtype
# strides
# shape
import numpy as np

# zeres(), ones(): 주어진 모양(shape)에 대해 모든 요소가 0 또는 1일 배열을 생성하는 함수
# 3행 4열
ones = np.ones((3,4))
print(ones)
print()

# zeros = np.zeros((3,4))
zeros = np.zeros((3,4), dtype = np.int16)
print(zeros)
print()
# 3차원 배열
zeros2 = np.zeros((2,3,4))
print(zeros2, zeros2.dtype)
print()

# arrange(): 지정된 범위 내에서 일정한 간격으로 숫자가 담긴 배열을 생성하는 함수
# range랑 범위가 비슷함
# a = np.arange(5)
# print(a,a.ndim,a.shape,a.size)
# a = np.arange(5,11,2)
# print(a,a.ndim,a.shape,a.size)
# a = np.arange(2.0, 11.8, 0.2)
# a = np.arange(2.0, 11.8, 0.2, dtype=np.int16)
a = np.arange(2.0, 11.8, 2, dtype=np.int16)
print(a,a.ndim,a.shape,a.size)


# linspace(): 지정된 범위 내에서 균등하게 분할된 숫자가 담긴 배열을 생성되는 함수
# reshape(): 배열의 모양(shape)을 변경하는 메서드로, 샐운 모양에 맞게 요소들을 재배열

# ndim
# dtype
# strides
# shape
# size: 배열의 총 요소 수, 모든 차원에 있는 요소의 수를 곱한 값
# ex) np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])의 size는 8입니다.
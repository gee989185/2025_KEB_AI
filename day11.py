import numpy as np

# v = np.array([1,3.9,-9,2])
# print(v,v.ndim)

# a = np.array([1,3,5,7])
# print(a)
# print(a,a.ndim)

b = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(b, b.ndim, b.dtype)
print()
print(b, b.ndim, b.dtype, b.strides)
print()

c = np.array([[[1,2,"3"], [4,5,6]],[[7,8,9],[10,11,12]]])
print(c,c.ndim, c.shape, c.dtype,c.strides)

# ndim: 배열의 차원 수(1차원, 2차원, 3차원: 백터, 행렬, 텐서 등의 차원 확인)
# dtype: 배열의 요소들의 데이터 타입
# strides: 메모리 상에서 다음 요소로 이동하기 위한 바이트 수
# ex) c >> (504, 252, 84): 첫 차원에서 다음 요소로 넘어갈때 504바이트를 이동해야함
# shape: 배열의 각 차원의 크기를 튜플형태로
# ex) [[1, 2, 3], [4, 5, 6]]의 shape은 (2, 3)입니다.

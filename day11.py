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


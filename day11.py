# Series
# DataFrame

import pandas as pd

# s = pd.Series([1,2,3,4])
s = pd.Series([1,2,3,4], index =['a','b','c','d'])
print(s)
print("-----------------")

s2 = pd.Series([99,100,98,91,92])
s2_subset = s2[1:4]
print(s2_subset)
print('----------------------')
s2_mean = s2.mean()
print(s2_mean)
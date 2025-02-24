# Series
# DataFrame

import pandas as pd

def square(n) -> int:
    return  n*n

df = pd.DataFrame(
    [[4, 7, 10],
     [5, 88, 11],
     [16, 9, 12]],
    index=[1, 2, 3],
    columns=['a', 'b', 'c']
)
print(df)
print('----------------')

# df2 = pd.melt(df)
# df2 = pd.melt(df).rename(columns={'variable':'var', 'value':'val'})
df2 = (pd.melt(df)
      .rename(columns={
                'variable':'var',
                 'value':'val'})
      .query('val >= 200')
).sort_values('val', ascending=False)
# 위에 저거 안됨

print(df2)
# df3 = df.loc[1:2]
# df3 = df.iloc[1:2]
df3 = df.iloc[:,[0,2]]
print(len(df3))
print("--------------")
print(df.apply(square))
print('---------------')
print(df.apply(lambda x: x*x))

# melt
# rename
# query
# sort.values
# loc
# iloc


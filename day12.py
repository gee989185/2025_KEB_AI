# Assginment
# v1.0) v0.9 파일의 사이킷런 라이브러리를 주석처리 하고 결측치를 산술평균으로 채워 넣으시오


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.DataFrame(
    {
        'A': [1,2,np.nan,4],
        'B': [np.nan,12,3,4],
        'C': [1,2,3,4],
    }
)


print(df)
print('------------')

# imputer =SimpleImputer(strategy="mean")
# df[['A','B']] = imputer.fit_transform(df[['A','B']])
df[['A','B']] = df[['A','B']].fillna(df[['A','B']].mean())
# mean = df[['A','B']].mean()
print(df)
print('------------')
print(mean(df))
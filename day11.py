#Assignment: v0.7) v0.6의 최근접이웃모델과 같이 동작하는 커스텀 클래스를 설계하시오.

#from statistics import LinearRegression

# 선형회귀모델
#from sklearn.linear_model import LinearRegression
# k-최근접 이웃 회귀 모델
#from sklearn.neighbors import KNeighborsRegressor

import pandas as pd
import matplotlib.pyplot as plt

#from tglearn import LinearRegression
from tglearn import KNeighborsRegressor

# 데이터를 다운로드하고 준비합니다.
ls = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")
# 2차원 리스트(독립변수, 명목변수 나누기)
X = ls[["GDP per capita (USD)"]].values
y = ls[["Life satisfaction"]].values
# print(ls)
# print(X)
# print(y)

# 데이터를 그래프로 나타냅니다
# 산점도(scatter)
# 격자무늬(grid)
# 범위 설정(axis
ls.plot(kind = 'scatter', grid=True,
        x ="GDP per capita (USD)", y = "Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# 선형 모델을 선택합니다.(1. 선형회귀모델, 2. k-최근접 이웃 회귀모델)
#model = LinearRegression()
model = KNeighborsRegressor(n_neighbors=3)
# 모델 훈련합니다.
model.fit(X,y)

# 키프로스에 대해 예측을 만듭니다.
print('------------')
X_new = [[31_721.3]]            # 2020년 대한민국 1인당 GDP
print(model.predict(X_new))     # 출력: [[6.30165767]]
# k-최근접 이웃회귀모델: 5.7
# 선형회귀모델: 5.89940454
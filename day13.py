# Assignment
# 데이터 로딩 -> 데이터 전처리 -> 타겟 및 독립변수 설정 -> 트레이닝/테스트 셋 설정 ->
# -> 모델 선택 및 학습 -> 예측 수행 -> 성능평가(rmse, mae) -> 시각화
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error ,r2_score

from day12 import y_test, y_pred

## 1. 데이터 로딩 및 데이터 확인(info(), head())
mpg = sns.load_dataset('mpg')
# print(mpg.info())
# print(mpg.head())
# print(mpg.describe()
# print(mpg[["horsepower"]].tail(10))


## 가정 설정: mpg(연비)와 큰 상관관계(음의 상과관계도)를 갖는 특징은 cylinders(-), displacement(-), horsepower(-), weight(-), acceleration(+), model_year(+)
# origin과 name은 큰 상관관계가 없을 것
X = mpg[["cylinders", "displacement", "horsepower", "weight",
              "acceleration", "model_year"]]    #"origin"과 "name", mpg를 제외한 Target
y = mpg["mpg"]      # mpg 독립변수

## 상관관계
corr_matrix = mpg.corr(numeric_only=True)
print(corr_matrix["mpg"].sort_values(ascending=False))
print(abs(corr_matrix["mpg"]).sort_values(ascending=False))
# 강한 상관관계(>=0.7) (-)weight > (-)displacement > (-)horsepower > (-)cylinders
# 상관관계 있음(>=0.4) model_year > acceleration



## 결측치 확인("horsepower") > 총 6개 > 영향력 x > dropna


## 데이터 분포 및 이상치 확인(히스토그램 확인 > 이상치 의심되는 특성 재확인)
mpg.hist(bins='auto', figsize=(12,8))
plt.show()

## 이상치 의심특성 displacement, model_year
print(mpg['displacement'].value_counts().sort_index())
print(mpg['model_year'].value_counts().sort_index())
# 이상치 없음


## 2. 데이터 전처리

## 결측치 처리
mpg.dropna(subset=["horsepower"], inplace= True)


## 특성 스케일링(min-max스케일링)
from sklearn.preprocessing import MinMaxScaler

minmax_scale = MinMaxScaler(feature_range=(-1,1))
X_min_max_scaled = minmax_scale.fit_transform(X)
mpg_scaled = pd.DataFrame(X_min_max_scaled, columns = X.columns)
print(mpg.head())


## 트레이닝/테스트셋 설정
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)


## 모델 선택 및 학습
model = LinearRegression()
model.fit(X_train, y_train)


## 예측 수행
y_pred = model.predict(X_test)


## 성능평가
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R-squared: {r2}')


## 시각화
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred)
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.title('MPG')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.show()
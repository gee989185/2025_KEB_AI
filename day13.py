# Assignment
# 데이터 로딩 -> 데이터 전처리 -> 타겟 및 독립변수 설정 -> 트레이닝/테스트 셋 설정 ->
# -> 모델 선택 및 학습 -> 예측 수행 -> 성능평가(rmse, mae) -> 시각화
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# from day12 import y_test, y_pred  # day12 import 제거 (불필요)

## 1. 데이터 로딩 및 데이터 확인(info(), head())
mpg = sns.load_dataset('mpg')
print(mpg.info())
print(mpg.head())
print(mpg.describe())

## 가정 설정: mpg(연비)와 큰 상관관계(음의 상과관계도)를 갖는 특징은 cylinders(-), displacement(-), horsepower(-), weight(-), acceleration(+), model_year(+)
# origin과 name은 큰 상관관계가 없을 것
X = mpg[["cylinders", "displacement", "horsepower", "weight",
          "acceleration", "model_year"]]  # "origin"과 "name", mpg를 제외한 Target
y = mpg["mpg"]  # mpg 독립변수

## 상관관계
corr_matrix = mpg.corr(numeric_only=True)
print(corr_matrix["mpg"].sort_values(ascending=False))
print(abs(corr_matrix["mpg"]).sort_values(ascending=False))
# 강한 상관관계(>=0.7) (-)weight > (-)displacement > (-)horsepower > (-)cylinders
# 상관관계 있음(>=0.4) model_year > acceleration

## 결측치 확인("horsepower") > 총 6개 > 영향력 x > dropna

## 데이터 분포 및 이상치 확인(히스토그램 확인 > 이상치 의심되는 특성 재확인)
mpg.hist(bins='auto', figsize=(12, 8))
plt.show()

## 이상치 의심특성 displacement, model_year
print(mpg['displacement'].value_counts().sort_index())
print(mpg['model_year'].value_counts().sort_index())
# 이상치 없음

## 2. 데이터 전처리

## 결측치 처리 (mpg에서 1차 제거)
mpg.dropna(subset=["horsepower"], inplace=True)
print(mpg.isnull().sum())

## 트레이닝/테스트셋 설정 (Train/Test 분할 먼저)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## **X_train에 NaN 값 직접 제거**
X_train.dropna(subset=["horsepower"], inplace=True)
y_train = y_train[X_train.index]
print("X_train NaN values after explicit dropna:")
print(X_train.isnull().sum())

## **X_test에 NaN 값 직접 제거 (추가)**
X_test.dropna(subset=["horsepower"], inplace=True)  # X_test에서 NaN 직접 제거
y_test = y_test[X_test.index]  # X_test index에 맞춰 y_test 업데이트 (중요!)
print("X_test NaN values after explicit dropna:")
print(X_test.isnull().sum())

## 특성 스케일링(min-max스케일링) (Train/Test 셋 분리 후 스케일링)
minmax_scale = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = minmax_scale.fit_transform(X_train)
X_test_scaled = minmax_scale.transform(X_test)

# 스케일링된 데이터를 DataFrame으로 변환 (optional)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

## 모델 선택 및 학습 (수정: 스케일링된 데이터 사용)
print("X_train_scaled NaN values before model fitting:")
print(X_train_scaled.isnull().sum())
model = LinearRegression()
model.fit(X_train_scaled, y_train)

## 예측 수행 (수정: 스케일링된 Test 데이터 사용)
print("X_test_scaled NaN values before prediction:")  # X_test_scaled NaN 값 확인 **(추가)**
print(X_test_scaled.isnull().sum())  # X_test_scaled NaN 값 확인 **(추가)**
y_pred = model.predict(X_test_scaled)

## 성능평가 (수정: squared 인자 제거)
rmse = mean_squared_error(y_test, y_pred) # squared 인자 제거
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R-squared: {r2}')

## 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.title('MPG')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.show()
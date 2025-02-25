# Assginment
# v1.1

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 데이터 로드
# titanic = sns.load_dataset('titanic')
# # print(type(titanic))
# print(titanic.info())
# #print(titanic.head())
#
# titanic["age"].value_counts()



titanic = sns.load_dataset('titanic')   # 데이터 로드
median_age = titanic['age'].median()  # 나이 중앙값 산출
titanic_fill_row = titanic.fillna({'age' : median_age})  # 결측치 처리

X = titanic_fill_row[['age']]  # 독립 변수 설정
#y = titanic_fill_row[['survived']]  # 종속 변수 설정
y = titanic_fill_row['survived']

# 데이터 분할 (학습 80%, 테스트 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성 및 훈련
model= KNeighborsClassifier(n_neighbors=5, metric='euclidean')
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 시각화
plt.figure(figsize=(6, 4))
plt.scatter(X_test, y_test, color='blue',alpha=0.2, label='Real')
plt.scatter(X_test, y_pred, color='red',alpha=0.2, label='Predicted')
plt.title('KNeighborsRegressor: Real vs Predicted')
plt.xlabel('Age')
plt.ylabel('Survivied')
plt.legend(loc='upper right')
plt.show()
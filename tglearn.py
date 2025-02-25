import numpy as np


class KNeighborsRegressor:
    def __init__(self,n_neighbors = 3):
        self.n_neighbors = n_neighbors
        self.X = None
        self.y = None

    def distance(self,X1,X2):
        return np.sqrt(np.sum((X1 - X2)**2))


    def fit(self,X,y):
        self.X = X
        self.y = y

    def predict(self,X_test):
        # 입력한 X값과 가장 가까운 distance를 가진 X값 추출(먼저 x값 저장 후 가장 가까운 n_neighbors 찾기)
        # 그 X값의 Y값들을 평균 내기
        y_predict =[]       # 예측값을 넣을 리스트(b)
        for i in X_test:
            distance_test =[]   # 입력 데이터와 다른나라GDP사이의 거리 리스트(a)
            for j in self.X:
                distance_test.append(self.distance(i,j))    # 저장(a)
            distance_test = np.array(distance_test)
            k_nearest = np.argsort(distance_test)[:self.n_neighbors]    #정렬 후 인덱스반환(n_neighbors까지)
            k_nearest_y = self.y[k_nearest]
            y_predict.append(np.mean(k_nearest_y))      # 저장(b)
            #print(distance_test)
            #print(k_nearest)                            # [5,6,4]
        return np.array(y_predict)



class LinearRegression:
    def __init__(self):
        self.slope = None  # weight
        self.intercept = None  # bias


    def fit(self, X, y):
        """
        learning function
        :param X: independent variable (2d array format)
        :param y: dependent variable (2d array format)
        :return: void
        """
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        denominator = np.sum(pow(X-X_mean, 2))
        numerator = np.sum((X-X_mean)*(y-y_mean))

        self.slope = numerator / denominator
        self.intercept = y_mean - (self.slope * X_mean)


    def predict(self, X) -> np.ndarray:
        """
        predict value for input
        :param X: new indepent variable
        :return: predict value for input (2d array format)
        """
        return self.slope * np.array(X) + self.intercept
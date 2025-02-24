# Assignment
# v0.7) v0.6의 최근접이웃모델과 같이 동작하는 커스텀 클래스를 설계하시오.

# from sklearn.linear_model import LinearRegression
# import tglearn as tg
# from tglearn import LinearRegression
# from tglearn import KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class customkneighbor:
    def __init__(self, n_neigh = 5):
        self.neigh = n_neigh
        self.X=None
        self.Y=None

    def fit(self,x,y):
        self.X = np.array(x)
        self.Y = np.array(y)
    
    def predict(self, X):
        predictions = []
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        for x in self.X:
            nearest_indices = np.argsort(distances)[:self.neigh]
            prediction = np.mean(self.y_train[nearest_indices])
            predictions.append(prediction)


ls = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")
X = ls[["GDP per capita (USD)"]].values
y = ls[["Life satisfaction"]].values

ls.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23500, 62500, 4, 9])
plt.show()

#model = LinearRegression()
model = KNeighborsRegressor()
model.fit(X, y)

X_new = [[31721.3]]  # ROK 2020
print(model.predict(X_new))
# LinearRegression 5.90
# KNeighborsRegressor 5.70
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

ls = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")
print(ls)
X = ls[["GDP per capita (USD)"]].values
Y = ls[["Life satisfaction"]].values

ls.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23500,62500,4,9])
plt.show()

model = LinearRegression()
model.fit(X,Y)
X_new = [[37655.2]]
print(model.predict(X_new))

model = KNeighborsRegressor()
model.fit(X,Y)
X_new = [[37655.2]]
print(model.predict(X_new))

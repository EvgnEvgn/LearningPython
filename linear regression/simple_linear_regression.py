import pandas as pd
import sklearn.linear_model as lm

x = pd.read_csv("X1.txt", sep=" ", header=None)[0]
y = pd.read_csv("Y1.txt", sep=" ", header=None)[0]


skm = lm.LinearRegression()
skm.fit(x, y)

print(skm.intercept_, skm.coef_)

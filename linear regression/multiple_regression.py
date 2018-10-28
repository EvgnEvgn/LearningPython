import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("MulRegressionData.txt", sep="\t")
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

corr = data.corr(method='pearson')
print()
print(corr.to_string())

skm = lm.LinearRegression()
skm.fit(x, y)
print("R coef =", skm.coef_)
print("Intercept =", skm.intercept_)


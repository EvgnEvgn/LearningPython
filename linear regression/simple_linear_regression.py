import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

x = pd.read_csv("X1.txt", sep=" ", header=None)
y = pd.read_csv("Y1.txt", sep=" ", header=None)
x = x.iloc[:]
y = y.iloc[:]

#x_test = x.iloc[80:]
#y_test = y.iloc[80:]

skm = lm.LinearRegression()
skm.fit(x, y)

print(skm.intercept_, skm.coef_)
plt.scatter(x, y, color='blue',)

plt.plot(x, skm.predict(x), color='red', linewidth=1)
plt.show()

import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np

data_x = pd.read_csv("X1.txt", sep=" ", header=None)
data_y = pd.read_csv("Y1.txt", sep=" ", header=None)
data_x = np.array(data_x.iloc[:])
data_y = np.array(data_y.iloc[:])

#x_test = x.iloc[80:]
#y_test = y.iloc[80:]

skm = lm.LinearRegression()
skm.fit(data_x, data_y)
print(skm.intercept_, skm.coef_)
plt.scatter(data_x, data_y, color='blue')

plt.plot(data_x, skm.predict(data_x), color='red', linewidth=1)
plt.show()
x = data_x.reshape((1, -1))
y = data_y.reshape((1, -1))

corr = np.corrcoef(x, y)
print('Correlation coefficient by numpy tools')
print(corr)
data = np.matrix([x[0], y[0]]).transpose()
corr = pd.DataFrame(data, columns=['X', 'Y']).corr(method="pearson")
print("Correlation coefficient by DataFrame tools")
print(corr)

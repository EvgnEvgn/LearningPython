import pandas as pd
import sklearn.linear_model as lm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from linear_regression.regression_helper import RegressionHelper


helper = RegressionHelper()
data_x = pd.read_csv("X1.txt", sep=" ", header=None)
data_y = pd.read_csv("Y1.txt", sep=" ", header=None)
data_x = np.array(data_x.iloc[:])
data_y = np.array(data_y.iloc[:])

x_train = data_x[:90]
y_train = data_y[:90]

x_test = data_x[90:]
y_test = data_y[90:]

# skm = lm.LinearRegression()
# skm.fit(x_train, y_train)
# predict_all = skm.predict(data_x)
# predicted = skm.predict(x_test)

x_train_with_const = sm.add_constant(x_train)
model = sm.OLS(y_train, x_train_with_const)
result = model.fit()
print(result.summary())
regression_params = result.params

y_predicted = result.predict(x_train_with_const)
x_test_with_const = sm.add_constant(x_test)
forecasting = result.predict(x_test_with_const)

dots, = plt.plot(x_train, y_train, "ro", color='blue')
line, = plt.plot(x_train, y_predicted, color='red', linewidth=1)

plt.legend([dots, line], ['Исходные данные', 'Линия регрессии'])

plt.xlabel('X_predictor')
plt.ylabel('Y_')

plt.show()

regressionLine, = plt.plot(x_train, y_predicted, color='red', linewidth=1)
predictedValues, = plt.plot(x_test, forecasting, "ro", color='red')
data, = plt.plot(x_test, y_test, "ro", color='blue')
plt.xlabel('X_predictor')
plt.ylabel('Y_')
plt.legend([regressionLine, predictedValues, data], ['Линия регрессии', 'Предсказанные значения', 'Истинные значения'])
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
print(helper.getRegressionEquation(regression_params))




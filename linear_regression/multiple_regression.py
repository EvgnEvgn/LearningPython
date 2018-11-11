import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv("MulRegressionData.txt", sep="\t")
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

x_train = x[:17]
y_train = y[:17]

x_test = x[17:]
y_test = y[17:]

corr = data.corr(method='pearson')
print('Correlation coefficients')
print(corr.to_string())

skm = lm.LinearRegression()
skm.fit(x_train, y_train)
print("\nR coef =", skm.coef_)
print("Intercept =", skm.intercept_)

#----------------------------------------------------------------
x_train_with_const = sm.add_constant(x_train)
model = sm.OLS(y_train, x_train_with_const)
results = model.fit()
x_test_with_const = sm.add_constant(x_test)
y_predicted = results.predict(x_test_with_const)
print(x_test)
print("Predicted values:")
print(y_predicted)

print('Model params by statsmodels')
print(results.params)
print(results.summary())

x_matrix = np.matrix(x_train_with_const.values)
y_matrix = np.matrix(y_train.values).transpose()
x_transposed = x_matrix.transpose()
x_mult_x_trans = np.dot(x_matrix, x_transposed)
inversion_x = np.linalg.inv(x_mult_x_trans)

print(len(x_transposed))
x_trans_multiply_y = np.dot(x_transposed, y_matrix)

result = np.dot(inversion_x, x_trans_multiply_y)


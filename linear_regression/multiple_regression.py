import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_excel("Folds5x2_pp.xlsx")
data = data.drop(['V', ], axis=1)
data = data.iloc[:100, :]
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

data_length = len(y)

numbers_of_train_samples = int(round(data_length*0.8))

x_train = x[:numbers_of_train_samples]
y_train = y[:numbers_of_train_samples]

x_test = x[numbers_of_train_samples:]
y_test = y[numbers_of_train_samples:]

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

result_of_prediction = pd.DataFrame(data=y_predicted, columns=['predicted'])
result_of_prediction = result_of_prediction.join(y_test)

print("Predicted values:")
print(result_of_prediction)

print('Model params by statsmodels')
print(results.params)
print(results.summary())

x_matrix = np.matrix(x_train_with_const.values)
y_matrix = np.matrix(y_train.values).transpose()
x_transposed = x_matrix.transpose()
x_mult_x_trans = np.dot(x_transposed, x_matrix)
inversion_x = np.linalg.inv(x_mult_x_trans)

res = np.dot(inversion_x, x_transposed)

result = np.dot(res, y_matrix)

print(result)

error = abs(y_predicted - y_test) / abs(y_test)
print(error)



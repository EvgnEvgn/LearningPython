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
x_ = sm.add_constant(x_train)
model = sm.OLS(y_train, x_)
results = model.fit()
x_test = sm.add_constant(x_test)
y_predicted = model.predict(x_test)
print("Predicted values:")
print(y_predicted)

print('Model params by statsmodels')
print(results.params)
print(results.summary())



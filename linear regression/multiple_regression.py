import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv("MulRegressionData.txt", sep="\t")
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

corr = data.corr(method='pearson')
print('Correlation coefficients')
print(corr.to_string())

skm = lm.LinearRegression()
skm.fit(x, y)
print("\nR coef =", skm.coef_)
print("Intercept =", skm.intercept_)

#----------------------------------------------------------------
x_ = sm.add_constant(x)
model = sm.OLS(y, x_)
results = model.fit()
#predicted = model.predict()
#print("Predicted values:")
#print(predicted)
#predictions = model.predict(x)
print('Model params by statsmodels')
print(results.params)
print(results.summary())

#print('Predicted values: ', predictions)

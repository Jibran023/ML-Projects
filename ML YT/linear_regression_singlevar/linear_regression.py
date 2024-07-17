import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Reading the excel file
df = pd.read_excel('Book1.xlsx')
print(df)

# Plotting
# plt.scatter(df.area, df.price, color='red', marker='+')
# plt.xlabel('Area(sqr ft)')
# plt.ylabel('Price(US $)')
# plt.title('Scatter plot of Area vs Price')
# plt.show() Uncomment to see the plot

# Implementing the Linear Regression
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price) # We are training the model using the available data points

# Predicting the price for the given area of 3300
predicted_price = reg.predict([[3300]])
print(predicted_price)

# we are using y = mx + b for linear regression
coeff_value = reg.coef_
print(coeff_value) # this is the coefficient value
intercept_value = reg.intercept_
print(intercept_value) # this is the intercept value

# doing y = mx + b manually
manual_value = 135.87867123*3300 + 180616.438835616432
print(manual_value)


# Plotting
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area, reg.predict(df[['area']]), color='blue')  # Ensure df[['area']] is 2D
plt.xlabel('Area (sqr ft)', fontsize=20)
plt.ylabel('Price (US $)', fontsize=20)
plt.title('Area vs Price with Linear Regression Line')
plt.show()
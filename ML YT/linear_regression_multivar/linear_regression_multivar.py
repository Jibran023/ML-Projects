import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math

# Reading the excel file
df = pd.read_excel('Book2.xlsx')
# print(df)


median_bedroom = math.floor(df.bedroom.median())
# print(median_bedroom)

df.bedroom = df.bedroom.fillna(median_bedroom) # this will fill NA values with the provided value
print(df)

# Ensure the 'area' column exists before fitting the model
if 'Area' in df.columns:
    reg = linear_model.LinearRegression()
    reg.fit(df[['Area', 'bedroom', 'age']], df['price'])  # Provide the independent values and the dependent value
    print("Model coefficients:", reg.coef_)
    print("Model intercept:", reg.intercept_)
    # price = m1xarea + m2xbedroom + m3xage + b
    # m1,m2,m3 are the coeffecients and b is the intercept
else:
    print("The 'Area' column does not exist in the DataFrame.")

# now we will predict the house prices for the following homes
# 1. 3000 sqft, 3 bedrooms, 40 years old
print(f"The price of the home would be: {reg.predict([[3000, 3, 40]])}")

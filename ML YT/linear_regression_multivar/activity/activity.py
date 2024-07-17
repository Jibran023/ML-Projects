import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
from word2number import w2n

# Reading the excel file
df = pd.read_excel('Book2.1.xlsx')
# print(df)

df.experience = df.experience.fillna('zero')
# print(df)


df['experience'] = df['experience'].apply(lambda x: w2n.word_to_num(x)) # Convert text experience to numeric


df.test_score = df.test_score.fillna(math.floor(df.test_score.median()))
# print(df)


reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score', 'interview_score']], df.salary)
# print("Model coefficients:", reg.coef_)
# print("Model intercept:", reg.intercept_)

# predicting the salaries for these candidates
# a) 2 yr exp, 9 test score, 6 interview score
# b) 12yr exp, 10 test score, 10 interview score

print(f"The salary of candidate A would be: {reg.predict([[2, 9, 6]])}")
print(f"The salary of candidate B would be: {reg.predict([[12, 10, 10]])}")


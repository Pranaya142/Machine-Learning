# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 22:15:37 2025

@author: komme
"""
# import libraries
import numpy as np


import matplotlib.pyplot as plt


import pandas as pd

# read the data
dataset= pd.read_csv(r'D:\GenAi\ML\Simple Linear Regression Model\Insurance_dataset\simplelinearregression.csv')

# devide into dependent variable and independent variable columns
x= dataset.iloc[:,:-1]

y= dataset.iloc[:,-1]

# test the data 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# select regression model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred= regressor.predict(x_test)

# plot the graph

plt.scatter(x_test,y_test,color= 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.xlabel('Age')
plt.ylabel('Premium')
plt.title('Insurance of a person')
plt.show()

m = regressor.coef_

c = regressor.intercept_

age_50= m* 50 +c

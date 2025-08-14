# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 23:00:07 2025

@author: komme
"""
# import libraries
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset= pd.read_csv(r'D:\GenAi\ML\Simple Linear Regression Model\Powerbill\Powerbill.csv')

# devide the table into x and y variables

x= dataset.iloc[:,1:-1]
y= dataset.iloc[:,-1]

# test the data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)

# select the model

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)

#predict values
y_pred= regressor.predict(x_test)

#plot the graph

plt.scatter(x_test,y_test,color = 'green')
plt.plot(x_train,regressor.predict(x_train),color = 'black')
plt.title('ElectricityBill')
plt.xlabel('Month')
plt.ylabel('Bill')
plt.show()

m = regressor.coef_
c= regressor.intercept_

m12= m * 12 + c
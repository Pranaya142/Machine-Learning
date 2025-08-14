# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 11:12:38 2025

@author: komme
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv(r'D:\sir gen Ai\august\ML\Simple linear regression\Salary_Data.csv')

x= dataset.iloc[:,:-1]
y= dataset.iloc[:,-1]


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_test,y_test, color= 'red')
plt.plot(x_train, regressor.predict(x_train),color= 'blue')
plt.xlabel('Year of experiance')
plt.ylabel('Salary')
plt.title('Salary vs Experiance ')
plt.show()

m = regressor.coef_

c = regressor.intercept_

exp_12 = m * 12 + c

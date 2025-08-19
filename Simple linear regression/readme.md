🪜 Steps:

    1.  import libraries with their alias names(Numpy , matplotlib.pyplot, pandas)
    2.  read the data (df = pd.read_csv(r''))
    3.  devide the data into dependent and independent using df.iloc[]
    4.  cleaned the data
    5.  test with 80-20 ratio.

Used Sklearn framework to work data preprocessing. 

Sklearn transformers:

    1.   .impute:  (from sklearn.impute import SimpleImputer)
            1.   used to fill the null values in the dataframe.
            2.   SimpleImputer is used for paramete tuning.(hyper perameter)
            3.   Fit is used to fill the null values.
            4.   Transform is update the fit values to data.
    2.    .preprocessing:  (from sklearn.preprocessing import LabelEncoder)

                 LabelEncoder is a function which is used to convert categorical data into numerics like 0,1,2,3,....

    3.    .model_selection: (from sklearn.model_selection import train_test_split)

           1.    train_test_split is used to perform testing ratio for the data.
           2.    test ratio is (80-20) or (70-30) or (75-25).
           3.    in this we have to pass random_state = 0 parameter,because with out this arguments every time when we run the code
                 we get alternate values in the dataset.with this machine cannot predict the values perfectly.
                 so we have to specify this "random_state=0" parameter.
           4.    in this code we can find x_train,x_test and y_train,y_test values.
           5.    while passing arguments into the train_test_split() function,no need to give both train_size and test_size,
                 one is enough to pass.machine will automatically understands remaining.
    
    4.   .linear_model:   (from sklearn.linear_model import LinearRegression)

          1.     with this transformer we build a model.
          2.     define a variable regressor and assign the LinearRegression() to it.
          3.     now with that parameter we can fit the x_train and y_train values.(regressor.fit(x_train,y_train))

    5.    predict the values:

                pass the x_test values to the regressor.predict function(ypred = regressor.predict(x_test))


To predict Future we sholud know slope(m) and constant(c) values.

use  m= regressor.coef_           (because in y=mx+c linear equation m is the coeffient of x)

c= regressor.intersept_      (c is the constant)



Used Pickle module for zip the model and save the model in the sysytem using import os & os.getcwd() function.

created front end for the model using Streamlit framework.

🪜 Steps:

    1.  import libraries with their alias names(Numpy , matplotlib.pyplot, pandas)
    2.  read the data (df = pd.read_csv(r''))
    3.  devide the data into dependent and independent using df.iloc[]
    4.  cleaned the data
    5.  testin with 80-20 ratio.

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
           5.    while passing arguments into the train_test_split() function,no need to give both train_size and test_size,one is enough to pass.machine                  will automatically understands remaining.
           

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

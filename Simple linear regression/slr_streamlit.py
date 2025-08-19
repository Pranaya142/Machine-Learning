import streamlit as slt
import pickle
import numpy as np


#load the model
with open(r'D:\GenAi\ML\Simple Linear Regression Model\simple_linear_regression_model\linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# set the title
slt.title("Salary Prediction app")

# add brief description
slt.write("This app predicts the salary")

# add input widget for user to enter year of experience
year_of_experience = slt.number_input("Enter Year of Experience", min_value=0.0, max_value=50.0, value=1.0,step=0.5)

# when the button is clicked make prediction
if slt.button("predict salary"):
    # make prediction using the trained model
    exp_input= np.array([[year_of_experience]])
    prediction = model.predict(exp_input)
    
    #display the result
    slt.success(f'The predicted salary for   {year_of_experience}  years of experience is   ${prediction[0]:,.2f}')
#disply the information
slt.write("The model was trained using dataset of salaries and year of experience built model by Pranaya")


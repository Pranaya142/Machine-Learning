import streamlit as str
import pickle
import numpy as np

with open(r'D:\GenAi\ML\Simple Linear Regression Model\Insurance_dataset\insurance_model.pkl', 'rb') as f:
    model = pickle.load(f)


str.title("Insurance app")

str.write("This is a simple insurance predict app")

Age_of_applicant = str.number_input("Enter the applicant age",min_value=0,max_value=100,value=1,step=1)

button = str.button("Predict Premium")
if button:
    Age_input = np.array([[Age_of_applicant]])
    premium = model.predict(Age_input)
    str.write(f"The predicted insurance premium for a {Age_of_applicant}-year-old is: {premium[0]:,.2f}")
str.write("This model was trained using a dataset of insurance premiums and ages.")

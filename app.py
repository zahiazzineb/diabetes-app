import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
try:
    model = joblib.load('random_forest.joblib')
except FileNotFoundError:
    st.error("Error: 'random_forest.joblib' not found. Please ensure the model file is in the same directory.")
    st.stop()

st.title("Diabetes Prediction App")
st.write("Enter the patient's details to predict the likelihood of diabetes.")

# Input features
st.sidebar.header("Patient Input Features")

def user_input_features():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("Glucose", 50, 200, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure", 20, 120, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 5, 70, 29)
    insulin = st.sidebar.slider("Insulin", 0, 900, 150) # Keep 0 as min as per original data before cleaning
    bmi = st.sidebar.slider("BMI", 15.0, 70.0, 33.0)
    diabetes_pedigree_function = st.sidebar.slider("Diabetes Pedigree Function", 0.078, 2.5, 0.472)
    age = st.sidebar.slider("Age", 21, 85, 30)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader("User Input Features")
st.write(df_input)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(df_input)
    prediction_proba = model.predict_proba(df_input)

    st.subheader("Prediction")
    if prediction[0] == 0:
        st.success("The patient is predicted to NOT have diabetes.")
    else:
        st.warning("The patient is predicted to have diabetes.")

    st.subheader("Prediction Probability")
    st.write(f"Probability of Not Diabetic (0): {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Diabetic (1): {prediction_proba[0][1]:.2f}")
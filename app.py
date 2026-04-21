import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and preprocessors
rf_model = joblib.load('random_forest_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
imputer_X = joblib.load('imputer_X.pkl')

st.set_page_config(layout="centered")

st.title("👨‍💻 Salary Prediction App")
st.write("### Enter the employee details to predict their salary")

# Input fields
with st.sidebar:
    st.header("Input Features")
    age = st.slider("Age", min_value=18, max_value=70, value=30, step=1)
    years_of_experience = st.slider("Years of Experience", min_value=0.0, max_value=40.0, value=5.0, step=0.5)

    # Gender
    gender_options = list(label_encoders['Gender'].classes_)
    # Filter out 'nan' if it exists and is not a valid user input choice
    if 'nan' in gender_options: gender_options.remove('nan')
    selected_gender_str = st.selectbox("Gender", options=gender_options)

    # Education Level
    education_options = list(label_encoders['Education Level'].classes_)
    if 'nan' in education_options: education_options.remove('nan')
    selected_education_str = st.selectbox("Education Level", options=education_options)

    # Job Title
    job_title_options = list(label_encoders['Job Title'].classes_)
    if 'nan' in job_title_options: job_title_options.remove('nan')
    selected_job_title_str = st.selectbox("Job Title", options=job_title_options, help="Select from the job titles the model was trained on.")

# Prepare input for prediction
if st.button("Predict Salary"):    
    # Encode categorical features
    encoded_gender = label_encoders['Gender'].transform([selected_gender_str])[0]
    encoded_education = label_encoders['Education Level'].transform([selected_education_str])[0]
    encoded_job_title = label_encoders['Job Title'].transform([selected_job_title_str])[0]

    # Create a DataFrame from inputs, matching the training data structure
    input_data = pd.DataFrame([[
        float(age),
        float(encoded_gender),
        float(encoded_education),
        float(encoded_job_title),
        float(years_of_experience)
    ]],
    columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

    # Impute missing values (although direct inputs, this ensures consistency with training data preprocessing)
    input_data_imputed = pd.DataFrame(imputer_X.transform(input_data), columns=input_data.columns)

    # Make prediction
    prediction = rf_model.predict(input_data_imputed)[0]

    st.success(f"Predicted Salary: ${prediction:,.2f}")

st.markdown("---")
st.write("**Note:** This prediction is based on a Random Forest Regressor model trained on the provided dataset. Factors not included in the model might influence actual salary.")

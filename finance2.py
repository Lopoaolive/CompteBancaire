import numpy as np
import streamlit as st 
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

#affiche des titres
st.title("prediction de la probabibilité d'acquisition de compte bancaire")
st.subheader("model de regression logistique")

# chargement du ficher joblib du model de regression logistique
model = joblib.load("final_modelreg.joblib")

# chargement des fichiers encoders des variables categorielle
location_type_encoder = joblib.load("location_type_encoder.joblib")
cellphone_access_encoder = joblib.load("cellphone_access_encoder.joblib")
gender_of_respondent_encoder = joblib.load("gender_of_respondent_encoder.joblib")
education_level_encoder = joblib.load("education_level_encoder.joblib")
job_type_encoder = joblib.load("job_type_encoder.joblib")


# Define the inference function
def inference(location_type, cellphone_access, age_of_respondent, gender_of_respondent, education_level, job_type):
    # Encodage des variables categorielle de l'inference
    new_data = np.array([
        location_type_encoder.transform([location_type]),
        cellphone_access_encoder.transform([cellphone_access]),
        np.array([age_of_respondent]),  # Convertion de la variable age en tableau
        gender_of_respondent_encoder.transform([gender_of_respondent]),
        education_level_encoder.transform([education_level]),
        job_type_encoder.transform([job_type])
    ])
    new_data_encoded = new_data.reshape(1, -1)  # Remodeler en tableau 2D 
    prediction = model.predict(new_data_encoded)
    return prediction



# saisie de l'utilisateur
location_type = st.selectbox("Location Type", options=["Urban", "Rural"])
cellphone_access = st.selectbox("Cellphone Access", options=["Yes", "No"])
age_of_respondent = st.number_input("Age of Respondent", min_value=15, max_value=100)
gender_of_respondent = st.selectbox("Gender of Respondent", options=["Male", "Female"])
education_level = st.selectbox("Education Level", options=["No formal education", "Primary education", "Secondary education", "Tertiary education", "Vocational/Specialised training", "Other/Dont know/RTA"])
job_type = st.selectbox("Job Type", options=["Farming and Fishing", "Self employed", "Formally employed Private", "Informally employed", "Formally employed Government", "Farming and Fishing", "Remittance Dependent", "Other Income", "No Income", "Dont Know/Refuse to answer"])

# Prediction
if st.button("Predict"):
    prediction = inference(location_type, cellphone_access, age_of_respondent, gender_of_respondent, education_level, job_type)
    if prediction == 1:
        st.success("Yes: La personne peut avoir compte bancaire.")
    else:
        st.success("No: La personne ne peut pas avoir de compte bancaire.")

import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Heart Stroke Prediction App", layout="centered")
st.title("🫀 Heart Stroke Prediction App")

st.write("Enter the details below to check stroke/heart risk.")

# ------------------------------------
# USER INPUT FIELDS (BASED ON YOUR DATASET)
# ------------------------------------

age = st.number_input(
    "Age",
    min_value=18, max_value=120, step=1
)

gender = st.selectbox(
    "Gender",
    ["male", "female", "other"]
)

hypertension = st.selectbox(
    "Hypertension (0 = No, 1 = Yes)",
    [0, 1]
)

cholesterol_level = st.number_input(
    "Cholesterol Level (mg/dL)",
    min_value=50, max_value=600, step=1
)

waist_circumference = st.number_input(
    "Waist Circumference (cm)",
    min_value=30, max_value=200, step=1
)

smoking_status = st.selectbox(
    "Smoking Status",
    ["never", "past", "current"]
)

physical_activity = st.selectbox(
    "Physical Activity",
    ["low", "moderate", "high"]
)

dietary_habits = st.selectbox(
    "Dietary Habits",
    ["healthy", "moderate", "unhealthy"]
)

air_pollution_exposure = st.selectbox(
    "Air Pollution Exposure",
    ["low", "moderate", "high"]
)

stress_level = st.selectbox(
    "Stress Level",
    ["low", "moderate", "high"]
)

fasting_blood_sugar = st.number_input(
    "Fasting Blood Sugar (mg/dL)",
    min_value=40.0, max_value=400.0, step=0.1
)

EKG_results = st.selectbox(
    "EKG Results",
    ["normal", "abnormal"]
)

medication_usage = st.selectbox(
    "Medication Usage (0 = No, 1 = Yes)",
    [0, 1]
)

# ------------------------------------
# PREDICT BUTTON
# ------------------------------------

if st.button("Predict Stroke Risk"):
    with open("lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Columns EXACTLY used during training
    columns = [
        'age', 'gender', 'hypertension', 'cholesterol_level',
        'waist_circumference', 'smoking_status', 'physical_activity',
        'dietary_habits', 'air_pollution_exposure', 'stress_level',
        'fasting_blood_sugar', 'EKG_results', 'medication_usage'
    ]

    # Build input row from user inputs
    sample_input = pd.DataFrame([[
        age,
        gender,
        hypertension,
        cholesterol_level,
        waist_circumference,
        smoking_status,
        physical_activity,
        dietary_habits,
        air_pollution_exposure,
        stress_level,
        fasting_blood_sugar,
        EKG_results,
        medication_usage
    ]], columns=columns)

    # Predict
    prediction = model.predict(sample_input)[0]

    # Meaningful output
    if prediction == 0:
        st.success("He is healthy.")
    else:
        st.error("You may get a heart stroke, consult a doctor as soon as possible.")

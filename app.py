"""
app.py — Obesity Level Prediction App
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import joblib
from src.train import ObesityPreprocessor

pipeline = joblib.load('models/pipeline.pkl')

LABEL_MAP = {0: 'Insufficient Weight', 1: 'Normal Weight', 2: 'Overweight', 4: 'Obesity'}

YES_NO   = {'Yes': 'yes', 'No': 'no'}
FREQ_MAP = {'No': 'no', 'Sometimes': 'Sometimes', 'Frequently': 'Frequently', 'Always': 'Always'}
MTRANS_MAP = {
    'Automobile':          'Automobile',
    'Bike':                'Bike',
    'Motorbike':           'Motorbike',
    'Public Transportation': 'Public_Transportation',
    'Walking':             'Walking',
}

st.title("Obesity Level Prediction")

# Inputs
gender         = st.selectbox("Gender", ['Male', 'Female'])
age            = st.number_input("Age (years)", min_value=1, max_value=120, value=25)
height         = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)
weight         = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.5)
family_history = st.selectbox("Family history of overweight?", ['Yes', 'No'])
favc           = st.selectbox("Do you eat high-calorie food frequently?", ['Yes', 'No'])

st.markdown("**Vegetable consumption frequency** · 1 = rarely · 2 = sometimes · 3 = daily")
fcvc = st.slider("Vegetable consumption frequency", 1, 3, value=2, step=1)

st.markdown("**Number of main meals per day**")
ncp = st.slider("Number of main meals", 1, 4, value=3, step=1)

caec_display = st.selectbox("Eating between meals?", list(FREQ_MAP.keys()))

smoke = st.selectbox("Do you smoke?", ['Yes', 'No'])

st.markdown("**Daily water consumption** · 1 = less than 1 L · 2 = 1–2 L · 3 = more than 2 L")
ch2o = st.slider("Daily water consumption (liters)", 1, 3, value=2, step=1)

scc = st.selectbox("Do you monitor your calorie intake?", ['Yes', 'No'])

st.markdown("**Physical activity frequency** · 0 = none · 1 = 1–2 days/wk · 2 = 2–4 days · 3 = 4–5 days")
faf = st.slider("Physical activity frequency", 0, 3, value=1, step=1)

st.markdown("**Daily technology use** · 0 = 0–2 h · 1 = 3–5 h · 2 = more than 5 h")
tue = st.slider("Time using technology devices", 0, 2, value=1, step=1)

calc_display   = st.selectbox("How often do you drink alcohol?", list(FREQ_MAP.keys()))
mtrans_display = st.selectbox("Primary transportation method", list(MTRANS_MAP.keys()))

# Validation 
errors = []
if height <= 0:
    errors.append("Height must be greater than 0.")
if age <= 0:
    errors.append("Age must be greater than 0.")
if weight <= 0:
    errors.append("Weight must be greater than 0.")

for msg in errors:
    st.warning(msg)

# Prediction 
if not errors and st.button("Predict"):
    input_df = pd.DataFrame([{
        'Gender':                         gender,
        'Age':                            float(age),
        'Height':                         float(height),
        'Weight':                         float(weight),
        'family_history_with_overweight': YES_NO[family_history],
        'FAVC':                           YES_NO[favc],
        'FCVC':                           float(fcvc),
        'NCP':                            float(ncp),
        'CAEC':                           FREQ_MAP[caec_display],
        'SMOKE':                          YES_NO[smoke],
        'CH2O':                           float(ch2o),
        'SCC':                            YES_NO[scc],
        'FAF':                            float(faf),
        'TUE':                            float(tue),
        'CALC':                           FREQ_MAP[calc_display],
        'MTRANS':                         MTRANS_MAP[mtrans_display],
    }])

    prediction = pipeline.predict(input_df)[0]
    label = LABEL_MAP.get(prediction, str(prediction))

    # colour-code the result by clinical severity
    color_fn = {0: st.info, 1: st.success, 2: st.warning, 4: st.error}
    color_fn.get(prediction, st.write)(f"**Predicted obesity level: {label}**")

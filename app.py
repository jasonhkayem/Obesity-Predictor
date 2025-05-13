
import streamlit as st
import pickle
import pandas as pd

with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('minmax_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("Obesity Prediction App")

#user input for fields
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.number_input("Age", min_value=1, max_value=120)
height = st.number_input("Height (in meters)", min_value=1.0, max_value=2.5)
weight = st.number_input("Weight (in kg)", min_value=10.0, max_value=300.0)
family_history = st.selectbox("Family history with overweight", ['Yes', 'No'])
favc = st.selectbox("Frequent consumption of high caloric food", ['Yes', 'No'])
fcvc = st.slider("Frequency of vegetable consumption", 1, 3, step=1)
st.markdown("""
**1**: Low or no vegetable consumption  
**2**: Moderate vegetable consumption  
**3**: Daily vegetable consumption
""")
ncp = st.slider("Number of main meals", 1, 4, step=1)
caec = st.selectbox("Consumption of food between meals", ['No', 'Sometimes', 'Frequently', 'Always'])
smoke = st.selectbox("Do you smoke?", ['Yes', 'No'])
ch2o = st.slider("Daily water consumption (liters)", 1.0, 3.0, step=0.1)
scc = st.selectbox("Do you monitor calories?", ['Yes', 'No'])
faf = st.slider("Physical activity frequency", 0, 3, step=1)
st.markdown("""
**1**: Low or no Physical activity  
**2**: Moderate Physical activity  
**3**: Daily Physical activity
""")
tue = st.slider("Time using technology devices", 0, 2, step=1)
st.markdown("""
**1**: Low or no Usage  
**2**: Moderate Usage  
**3**: Daily Usage
""")
calc = st.selectbox("Alcohol consumption", ['No', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.selectbox("Transportation method", ['Public Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

#use the same encoding as during training
input_data = pd.DataFrame([[
    gender, age, height, weight, family_history, favc, fcvc, ncp, caec,
    smoke, ch2o, scc, faf, tue, calc, mtrans
]], columns=[
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'
])

# Apply the same preprocessing as during training
input_data_encoded = pd.get_dummies(input_data)

# Ensure the input columns match the training data
# Load original training columns for alignment
training_data = pd.read_csv('Obesity Data Set.csv')
training_columns = pd.get_dummies(training_data.drop(columns=['NObeyesdad'])).columns
input_data_encoded = input_data_encoded.reindex(columns=training_columns, fill_value=0)

# Scale the input
scaled_input = scaler.transform(input_data_encoded)

if st.button("Predict"):
    prediction = model.predict(scaled_input)
    predicted_label = prediction[0]

    #labels mapping
    label_map = {
        'Insufficient_Weight': 'Insufficient weight',
        'Normal_Weight': 'Normal weight',
        'Overweight_Level_I': 'Overweight',
        'Overweight_Level_II': 'Overweight',
        'Obesity_Type_I': 'Obese',
        'Obesity_Type_II': 'Obese',
        'Obesity_Type_III': 'Obese'
    }

    readable_label = label_map.get(predicted_label, predicted_label)

    st.success(f"Prediction: {readable_label}")
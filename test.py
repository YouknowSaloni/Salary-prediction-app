import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model, scaler, and encoders
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar input
st.sidebar.header("Input Employee Details")

# Input fields
age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Workclass", encoders["workclass"].classes_)
fnlwgt = st.sidebar.number_input("fnlwgt (e.g. 100000)", value=100000)
educational_num = st.sidebar.slider("Educational Number", 5, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", encoders["marital-status"].classes_)
occupation = st.sidebar.selectbox("Occupation", encoders["occupation"].classes_)
relationship = st.sidebar.selectbox("Relationship", encoders["relationship"].classes_)
race = st.sidebar.selectbox("Race", encoders["race"].classes_)
gender = st.sidebar.selectbox("Gender", encoders["gender"].classes_)
capital_gain = st.sidebar.number_input("Capital Gain", value=0)
capital_loss = st.sidebar.number_input("Capital Loss", value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
native_country = st.sidebar.selectbox("Native Country", encoders["native-country"].classes_)

# Build input DataFrame
input_dict = {
    'age': age,
    'workclass': encoders['workclass'].transform([workclass])[0],
    'fnlwgt': fnlwgt,
    'educational-num': educational_num,
    'marital-status': encoders['marital-status'].transform([marital_status])[0],
    'occupation': encoders['occupation'].transform([occupation])[0],
    'relationship': encoders['relationship'].transform([relationship])[0],
    'race': encoders['race'].transform([race])[0],
    'gender': encoders['gender'].transform([gender])[0],
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': encoders['native-country'].transform([native_country])[0]
}

input_df = pd.DataFrame([input_dict])

st.write("### 🔎 Input Data")
st.write(input_df)

# Prediction
if st.button("Predict Salary Class"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    
    st.write(f"🔢 Raw model prediction output: {prediction}")
    st.success(f"✅ Predicted Salary Class: **{prediction[0]}**")



# --- Batch Prediction ---
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:", df.head())

        # Encode categorical columns using saved encoders
        for col in encoders:
            if col in df.columns:
                df[col] = encoders[col].transform(df[col])

        # Scale features
        df_scaled = scaler.transform(df)
        predictions = model.predict(df_scaled)
        df["PredictedIncomeClass"] = np.where(predictions == 1, ">50K", "≤50K")

        st.write("✅ Predictions:")
        st.write(df.head())

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV with Predictions", csv, file_name='predicted_salaries.csv')

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

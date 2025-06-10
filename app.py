import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
from streamlit_lottie import st_lottie
import requests

# --- Load model ---
model = joblib.load("model.pkl")

# --- Helper: Load Lottie Animation ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- Helper: Generate PDF Report ---
def create_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Diabetes Risk Report", ln=True, align="C")
    pdf.ln(10)

    for key, value in data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# --- Page Setup ---
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

# --- Dark Mode Styling (optional tweak) ---
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    label, input, div[data-testid="stNumberInput"] {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Logo + Lottie Animation ---
st.image("https://i.imgur.com/hW4Qvz5.png", width=80)
lottie = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_u4yrau.json")
if lottie:
    st_lottie(lottie, height=200)

# --- Title ---
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Diabetes Risk Prediction App</h1>", unsafe_allow_html=True)
st.markdown("### Enter your health parameters below:")

# --- Input Form ---
pregnancies = st.number_input("Pregnancies", 0)
glucose = st.number_input("Glucose Level", 0)
bp = st.number_input("Blood Pressure", 0)
skin = st.number_input("Skin Thickness", 0)
insulin = st.number_input("Insulin Level", 0)
bmi = st.number_input("BMI", 0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0)
age = st.number_input("Age", 0)

# --- Predict Button ---
if st.button("Predict"):
    data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(data)
    result = "Diabetes Risk: YES" if prediction[0] == 1 else "Diabetes Risk: NO"
    st.success(result)

    # --- Report Data ---
    user_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
        "Prediction": "Diabetes" if prediction[0] == 1 else "No Diabetes"
    }

    df_result = pd.DataFrame([user_data])

    # --- Chart ---
    st.subheader("📊 Health Overview")
    fig, ax = plt.subplots()
    params = ['Glucose', 'BloodPressure', 'BMI', 'Age']
    values = [glucose, bp, bmi, age]
    ax.bar(params, values, color='#2E86C1')
    ax.set_title("Health Parameters Overview")
    ax.set_ylabel("Value")
    st.pyplot(fig)

    # --- CSV Download ---
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download CSV Report",
        data=csv,
        file_name='diabetes_report.csv',
        mime='text/csv'
    )

    # --- PDF Download ---
    pdf_buffer = create_pdf(user_data)
    st.download_button("📄 Download PDF Report", data=pdf_buffer, file_name="diabetes_report.pdf", mime="application/pdf")

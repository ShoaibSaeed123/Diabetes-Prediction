import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

# Sample input: [Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]
data = np.array([[1, 120, 70, 20, 80, 25.6, 0.5, 31]])

# Predict
prediction = model.predict(data)
print("Diabetes Risk:", "Yes" if prediction[0] == 1 else "No")

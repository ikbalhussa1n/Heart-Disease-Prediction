
import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
@st.cache_resource
def load_resources():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

model, scaler = load_resources()

st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")

st.title("Heart Disease Prediction")
st.write("This app predicts the likelihood of heart disease based on your input features.")

st.sidebar.header("Input Features")

# Input fields for features
age = st.sidebar.slider("Age", min_value=29, max_value=77, value=50)
sex = st.sidebar.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], format_func=lambda x:
    {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-Anginal Pain",
        3: "Asymptomatic"
    }.get(x))
thalach = st.sidebar.slider("Maximum Heart Rate Achieved (thalach)", min_value=71, max_value=202, value=150)
exang = st.sidebar.radio("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2], format_func=lambda x:
    {
        0: "Upsloping",
        1: "Flat",
        2: "Downsloping"
    }.get(x))
ca = st.sidebar.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thallium Stress Test Result (thal)", options=[0, 1, 2, 3], format_func=lambda x:
    {
        0: "Unknown", # Based on unique values [1, 2, 3, 0]
        1: "Normal",
        2: "Fixed Defect",
        3: "Reversible Defect"
    }.get(x))


# Create a dictionary for user inputs
user_data = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}

# Convert user input to a numpy array
features = np.array(list(user_data.values())).reshape(1, -1)

if st.button("Predict"): # Prediction button
    # Scale the input features
    scaled_features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("Heart Disease Detected")
        st.write(f"Probability of Heart Disease: {prediction_proba[0, 1]*100:.2f}%")
    else:
        st.success("No Heart Disease")
        st.write(f"Probability of No Heart Disease: {prediction_proba[0, 0]*100:.2f}%")

st.markdown("---")
st.info("Disclaimer: This is a predictive model for educational purposes and should not be used for medical diagnosis.")

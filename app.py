import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load model and scaler
model = load_model(r"C:\\Users\\reddy\\Downloads\\autoencoder_model.h5", compile=False)
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = np.load("C:\\Users\\reddy\\Downloads\\scaler_params.npy", allow_pickle=True)

st.title("🧠 Anomaly Detection with Autoencoder")

# Input form for 10 features
st.markdown("### ✍️ Input 10 Features")
inputs = []
cols = st.columns(5)
for i in range(10):
    with cols[i % 5]:
        val = st.number_input(f"Feature {i}", key=f"feature_{i}")
        inputs.append(val)

# Prediction
if st.button("🔍 Detect Anomaly"):
    input_array = np.array([inputs])
    scaled_input = scaler.transform(input_array)
    
    reconstructed = model.predict(scaled_input)
    mse = np.mean(np.power(scaled_input - reconstructed, 2), axis=1)
    
    # Replace with your real threshold
    threshold = 0.02
    is_anomaly = mse[0] > threshold

    st.markdown("---")
    st.subheader("🔎 Prediction Result")
    st.write(f"Reconstruction Error (MSE): **{mse[0]:.5f}**")
    if is_anomaly:
        st.error("❌ This is likely an **Anomaly**.")
    else:
        st.success("✅ This is likely **Normal**.")

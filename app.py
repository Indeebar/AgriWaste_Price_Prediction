# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 11:17:14 2025

@author: Indeebar Ray
"""

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load the trained model and encoders
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("price_model.h5")
    return model

@st.cache_resource
def load_encoders():
    waste_encoder = joblib.load("waste_encoder.pkl")
    demand_encoder = joblib.load("demand_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    return waste_encoder, demand_encoder, scaler

# Load everything
model = load_model()
waste_encoder, demand_encoder, scaler = load_encoders()

# Streamlit App UI
st.title("Agricultural Waste Price Prediction")
st.write("This app predicts the market price per kg of agricultural waste based on quantity and demand level.")

# Waste Type Selection
st.subheader("Select Waste Type")
waste_types = waste_encoder.classes_
waste_type = st.selectbox("Choose a waste type:", waste_types)

# Quantity Input
quantity = st.number_input("Enter quantity in kg:", min_value=1, step=1)

# Demand Level Selection
st.subheader("Select Demand Level")
demand_levels = ["Low", "Medium", "High"]
demand_level = st.selectbox("Choose a demand level:", demand_levels)

# Prediction Button
if st.button("Predict Price"):
    waste_encoded = waste_encoder.transform([waste_type])[0]
    demand_encoded = demand_encoder.transform([demand_level])[0]
    quantity_scaled = scaler.transform([[quantity]])[0][0]

    predicted_price_per_kg = model.predict(np.array([[quantity_scaled, demand_encoded, waste_encoded]]))[0][0]
    total_price = predicted_price_per_kg * quantity

    st.subheader("Predicted Results")
    st.write(f"**Predicted Price per kg:** ₹{predicted_price_per_kg:.2f}")
    st.write(f"**Total Predicted Price:** ₹{total_price:.2f}")

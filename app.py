import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

@st.cache_resource
def load_data():
    data = pd.read_csv("Cars_Data.csv")
    return data

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboost_model.cbm")
    return model

data = load_data()
model = load_model()

# App UI
st.title("Used Car Price Predictor")
st.image("carwow-shutterstock_2356848413.jpg")

# Create mappings from original data
brands = data["brand"].unique().tolist()
models = data["model"].unique().tolist()
colors = data["color"].unique().tolist()
transmissions = data["transmission_type"].unique().tolist()
fuel_types = data["fuel_type"].unique().tolist()

# Sidebar inputs
st.sidebar.header("Car Specifications")
brand = st.sidebar.selectbox("Brand", brands)
model_name = st.sidebar.selectbox("Model", models)
color = st.sidebar.selectbox("Color", colors)
transmission = st.sidebar.selectbox("Transmission", transmissions)
fuel_type = st.sidebar.selectbox("Fuel Type", fuel_types)
power_ps = st.sidebar.number_input("Power (PS)", min_value=50, value=150)
mileage = st.sidebar.number_input("Mileage (km)", min_value=0, value=50000)
vehicle_age = st.sidebar.number_input("Vehicle Age (years)", min_value=0, value=5)

# Create input DataFrame
input_data = {
    "power_kw": power_kw,
    "power_ps": power_ps,
    "mileage_in_km": mileage,
    "vehicle_age": vehicle_age,
    "brand": brands.index(brand),
    "model": models.index(model_name),
    "color": colors.index(color),
    "transmission_type": transmissions.index(transmission),
    "fuel_type": fuel_types.index(fuel_type)
}
input_df = pd.DataFrame([input_data])
input_df = input_df[["power_kw", "power_ps", "mileage_in_km", "vehicle_age", "brand", "model", "color", "transmission_type", "fuel_type"]]

# Prediction
if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.subheader(f"Predicted Value: ${prediction:,.2f}")
    st.balloons()

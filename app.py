import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostRegressor

# -------------------------------
# Caching functions to load artifacts
# -------------------------------
@st.cache_resource
def load_data():
    data = pd.read_csv("Cars_Data.csv")
    return data

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboost_model.cbm")
    return model

@st.cache_resource
def load_expected_columns():
    # expected_columns.pkl should contain the list of feature names after get_dummies
    return joblib.load("expected_columns.pkl")

@st.cache_resource
def load_scaler():
    # scaler.pkl is the fitted StandardScaler used on numerical features during training
    return joblib.load("scaler.pkl")

# Load data, model, and preprocessing artifacts
data = load_data()
model = load_model()
expected_columns = load_expected_columns()
scaler = load_scaler()

# App Title and Image
st.title("Used Car Price Predictor")
st.image("carwow-shutterstock_2356848413.jpg")

# Create mappings for dropdown selections from the original data
brands = data["brand"].unique().tolist()
models_list = data["model"].unique().tolist()  # renamed to avoid conflict with model variable
colors = data["color"].unique().tolist()
transmissions = data["transmission_type"].unique().tolist()
fuel_types = data["fuel_type"].unique().tolist()

# Sidebar Inputs
st.sidebar.header("Car Specifications")

brand_input = st.sidebar.selectbox("Brand", brands)
model_input = st.sidebar.selectbox("Model", models_list)
color_input = st.sidebar.selectbox("Color", colors)
transmission_input = st.sidebar.selectbox("Transmission", transmissions)
fuel_type_input = st.sidebar.selectbox("Fuel Type", fuel_types)
power_ps = st.sidebar.number_input("Power (PS)", min_value=50, value=150)
power_kw = st.sidebar.number_input("Power (KW)", min_value=50 * 0.7355, value=150 * 0.7355)
mileage = st.sidebar.number_input("Mileage (km)", min_value=0, value=50000)
vehicle_age = st.sidebar.number_input("Vehicle Age (years)", min_value=0, value=5)

# Create a Raw Input DataFrame
# -------------------------------
# Note: During training, your raw data had these columns (with categorical variables as strings).
raw_input = pd.DataFrame([{
    "power_kw": power_kw,
    "power_ps": power_ps,
    "mileage_in_km": mileage,
    "vehicle_age": vehicle_age,
    "brand": brand_input,
    "model": model_input,
    "color": color_input,
    "transmission_type": transmission_input,
    "fuel_type": fuel_type_input
}])

# Preprocessing: One-Hot Encoding for Categorical Features
categorical_columns = ["brand", "model", "color", "transmission_type", "fuel_type"]
input_dummies = pd.get_dummies(raw_input, columns=categorical_columns, drop_first=True)

# Align Input DataFrame to the Expected Columns
# -------------------------------
# Reindex to match the training features (fill missing columns with 0)
input_df = input_dummies.reindex(columns=expected_columns, fill_value=0)

# Scale Numerical Features
numerical_columns = ["power_kw", "power_ps", "mileage_in_km", "vehicle_age"]
input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

# (Optional) Display the processed input for debugging
st.write("Processed Input Data:")
st.write(input_df)

# Prediction
if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.subheader(f"Predicted Value: ${prediction:,.2f}")
    st.balloons()

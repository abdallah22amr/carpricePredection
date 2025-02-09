import streamlit as st
import pandas as pd
import joblib
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

@st.cache_resource
def load_expected_columns():
    return joblib.load("expected_columns.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

# -------------------------------
# Load Data, Model, and Preprocessing Artifacts
# -------------------------------
data = load_data()
model = load_model()
expected_columns = load_expected_columns()
scaler = load_scaler()


# Custom CSS injection
st.markdown(
    """
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global styles */
    body {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #1e1e1e, #2e2e2e) !important;
        color: #e0e0e0 !important;
    }
    .main-container {
        padding: 2rem 1rem;
    }
    /* Container for input fields */
    .input-container {
        background: #121212;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    /* Input widget styling */
    .stSelectbox, .stNumberInput, .stTextInput {
        border-radius: 8px !important;
        background-color: #2e2e2e !important;
        border: 1px solid #444 !important;
        color: #e0e0e0 !important;
    }
    /* Modern button styling */
    .stButton>button {
        background: linear-gradient(135deg, #4CAF50, #45A049) !important;
        color: white !important;
        font-weight: bold !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        transition: transform 0.3s ease;
        border: none !important;
        cursor: pointer !important;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    /* Prediction card styling */
    .prediction-card {
        background: #222;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        font-size: 28px;
        font-weight: 600;
        color: #4CAF50;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        margin-top: 20px;
    }
    /* Image styling */
    .header-image {
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    /* Footer styling */
    .footer {
        text-align: center;
        color: #777;
        margin-top: 40px;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and header image
st.title("Used Car Price Predictor")
st.image("carwow-shutterstock_2356848413.jpg", use_container_width=True)

# Instead of a sidebar, we create an input container in the main area
with st.container():
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #e0e0e0;'>Enter Car Specifications</h3>", unsafe_allow_html=True)

# Create Dropdown Mappings from Original Data
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
fuel_consumption = st.sidebar.number_input("Fuel Consumption (L/100km)", min_value=0.0, value=8.0)

# Create Raw Input DataFrame
raw_input = pd.DataFrame([{
    "power_kw": power_kw,
    "power_ps": power_ps,
    "fuel_consumption_l_100km.1": fuel_consumption,
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

# Reindex the DataFrame to match the exact order and names from training.
input_df = input_dummies.reindex(columns=expected_columns, fill_value=0)

# Scale Numerical Features
numerical_columns = ["power_kw", "power_ps", "fuel_consumption_l_100km.1", "mileage_in_km", "vehicle_age"]
input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

# (Optional) Display the processed input for debugging
# st.write("Processed Input Data:")
# st.write(input_df)

# Display input summary in a container
with st.container():
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    st.markdown("<h4>Car Specifications</h4>", unsafe_allow_html=True)
    st.dataframe(input_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction
if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.subheader(f"Predicted Value: ${prediction:,.2f}")
    st.markdown(f"<div class='prediction-card'>Predicted Price: <br> ${predicted_price:,.2f}</div>", unsafe_allow_html=True)
    st.balloons()
    
# Footer
st.markdown("<div class='footer'>Modern Car Price Predictor App © 2025</div>", unsafe_allow_html=True)

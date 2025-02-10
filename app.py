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

data = load_data()
model = load_model()
expected_columns = load_expected_columns()
scaler = load_scaler()

# Custom CSS Injection (All rules end with !important)
import streamlit as st

st.markdown(
    """
    <style>
    :root {
         color-scheme: dark !important;
    }
    /* Global Background Image */
    [data-testid="stAppViewContainer"], .stApp {
         background: url('https://www.motorfinanceonline.com/wp-content/uploads/sites/6/2025/01/carwow-shutterstock_2356848413.jpg') no-repeat center center fixed !important;
         background-size: cover !important;
         position: relative !important;
         z-index: 0 !important;
    }
    /* Dim the background */
    [data-testid="stAppViewContainer"]::before {
         content: "" !important;
         position: absolute !important;
         top: 0 !important;
         left: 0 !important;
         right: 0 !important;
         bottom: 0 !important;
         background: rgba(0, 0, 0, 0.7) !important;
         z-index: -1 !important;
    }
    /* Force dark mode on body */
    body {
         background-color: #1e1e1e !important;
         color: #e0e0e0 !important;
         font-family: 'Inter', sans-serif !important;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
         background: rgba(0, 0, 0, 0.7) !important;
         color: #e0e0e0 !important;
    }
    /* Sidebar toggle arrow styling */
    [data-testid="stSidebarNav"] {
         display: block !important;
         background: #121212 !important;
         border: none !important;
    }
    [data-testid="stSidebarNav"] svg {
         fill: #e0e0e0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Used Car Price Predictor")

# Input Fields in Main Area
st.markdown("### Enter Car Specifications")
col1, col2 = st.columns(2)
with col1:
    brand_input = st.selectbox("Brand", data["brand"].unique().tolist())
    model_input = st.selectbox("Model", data["model"].unique().tolist())
    color_input = st.selectbox("Color", data["color"].unique().tolist())
    transmission_input = st.selectbox("Transmission", data["transmission_type"].unique().tolist())
    fuel_type_input = st.selectbox("Fuel Type", data["fuel_type"].unique().tolist())
with col2:
    power_ps = st.number_input("Power (PS)", min_value=50, value=150)
    power_kw = st.number_input("Power (KW)", min_value=int(50 * 0.7355), value=int(150 * 0.7355))
    mileage = st.number_input("Mileage (km)", min_value=0, value=50000)
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, value=5)
    fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=0.0, value=8.0)

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

categorical_columns = ["brand", "model", "color", "transmission_type", "fuel_type"]
input_dummies = pd.get_dummies(raw_input, columns=categorical_columns, drop_first=True)
input_df = input_dummies.reindex(columns=expected_columns, fill_value=0)

numerical_columns = ["power_kw", "power_ps", "fuel_consumption_l_100km.1", "mileage_in_km", "vehicle_age"]
input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

with st.sidebar:
    st.markdown("## Car Specifications Summary")
    spec_summary = {
        "Brand": brand_input,
        "Model": model_input,
        "Color": color_input,
        "Transmission": transmission_input,
        "Fuel Type": fuel_type_input,
        "Power (PS)": power_ps,
        "Power (KW)": power_kw,
        "Mileage (km)": mileage,
        "Vehicle Age": vehicle_age,
        "Fuel Consumption (L/100km)": fuel_consumption,
    }
    for key, value in spec_summary.items():
        st.write(f"**{key}:** {value}")

if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.markdown(f"<div class='prediction-card'>Predicted Value: ${prediction:,.2f}</div>", unsafe_allow_html=True)
    st.balloons()

st.markdown("<div class='footer'>Modern Car Price Predictor App © 2025</div>", unsafe_allow_html=True)

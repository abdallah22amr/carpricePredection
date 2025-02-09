import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostRegressor

# -------------------------------
# Cached Functions to Load Artifacts
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
    # expected_columns.pkl contains the list of feature names from training (after get_dummies)
    return joblib.load("expected_columns.pkl")

@st.cache_resource
def load_scaler():
    # scaler.pkl is the fitted StandardScaler from training
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
    .reportview-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f4f6f9;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 2px solid #e0e0e0;
    }
    button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Used Car Price Predictor")
st.image("carwow-shutterstock_2356848413.jpg")

# Main content area using columns
col1, col2 = st.columns(2)
with col1:
    st.subheader("Input Summary")
    st.write(f"**Brand:** {brand}")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Color:** {color}")
    st.write(f"**Transmission:** {transmission}")
    st.write(f"**Fuel Type:** {fuel_type}")
with col2:
    st.subheader("Technical Specs")
    st.write(f"**Power (PS):** {power_ps}")
    st.write(f"**Power (KW):** {power_kw:.2f}")
    st.write(f"**Mileage (km):** {mileage}")
    st.write(f"**Vehicle Age:** {vehicle_age}")


# -------------------------------
# Create Dropdown Mappings from Original Data
# -------------------------------
brands = data["brand"].unique().tolist()
models_list = data["model"].unique().tolist()  # renamed to avoid conflict with model variable
colors = data["color"].unique().tolist()
transmissions = data["transmission_type"].unique().tolist()
fuel_types = data["fuel_type"].unique().tolist()

# -------------------------------
# Sidebar Inputs
# -------------------------------
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
# Added missing input for fuel consumption
fuel_consumption = st.sidebar.number_input("Fuel Consumption (L/100km)", min_value=0.0, value=8.0)

# -------------------------------
# Create Raw Input DataFrame
# -------------------------------
# The keys here should match the column names in your training data prior to one-hot encoding.
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

# -------------------------------
# Preprocessing: One-Hot Encoding for Categorical Features
# -------------------------------
categorical_columns = ["brand", "model", "color", "transmission_type", "fuel_type"]
input_dummies = pd.get_dummies(raw_input, columns=categorical_columns, drop_first=True)

# -------------------------------
# Align Input Data with Expected Feature Columns
# -------------------------------
# Reindex the DataFrame to match the exact order and names from training.
input_df = input_dummies.reindex(columns=expected_columns, fill_value=0)

# -------------------------------
# Scale Numerical Features
# -------------------------------
# List of numerical features as used during training.
numerical_columns = ["power_kw", "power_ps", "fuel_consumption_l_100km.1", "mileage_in_km", "vehicle_age"]
input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

# (Optional) Display the processed input for debugging
# st.write("Processed Input Data:")
# st.write(input_df)

# -------------------------------
# Prediction
# -------------------------------
if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.subheader(f"Predicted Value: ${prediction:,.2f}")
    st.balloons()

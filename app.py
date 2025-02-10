import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostRegressor

# -------------------------------------------
# Caching Functions to Load Data, Model, and Artifacts
# -------------------------------------------
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

# Load artifacts
data = load_data()
model = load_model()
expected_columns = load_expected_columns()
scaler = load_scaler()

# -------------------------------------------
# Custom CSS Injection for a Modern Look
# -------------------------------------------
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

# -------------------------------------------
# App Title and Header Image
# -------------------------------------------
st.title("Used Car Price Predictor")
st.image("carwow-shutterstock_2356848413.jpg", use_container_width=True)

# -------------------------------------------
# Create Dropdown Mappings from Original Data
# -------------------------------------------
brands = data["brand"].unique().tolist()
models_list = data["model"].unique().tolist()  # Renamed to avoid conflict with 'model'
colors = data["color"].unique().tolist()
transmissions = data["transmission_type"].unique().tolist()
fuel_types = data["fuel_type"].unique().tolist()

# -------------------------------------------
# Sidebar Inputs for Car Specifications
# -------------------------------------------
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

# -------------------------------------------
# Create Raw Input DataFrame
# -------------------------------------------
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

# -------------------------------------------
# Preprocessing: One-Hot Encoding & Scaling
# -------------------------------------------
categorical_columns = ["brand", "model", "color", "transmission_type", "fuel_type"]
input_dummies = pd.get_dummies(raw_input, columns=categorical_columns, drop_first=True)
# Reindex to match training features
input_df = input_dummies.reindex(columns=expected_columns, fill_value=0)
# Scale numerical features
numerical_columns = ["power_kw", "power_ps", "fuel_consumption_l_100km.1", "mileage_in_km", "vehicle_age"]
input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

# -------------------------------------------
# Display Input Summary in a Container (Three Columns)
# -------------------------------------------
with st.container():
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    st.markdown("<h4>Car Specifications</h4>", unsafe_allow_html=True)
    
    # Use raw_input for display (preserving original text values)
    input_items = list(raw_input.iloc[0].items())
    
    # Calculate splitting indices to divide items into three columns evenly
    n = len(input_items)
    base = n // 3
    r = n % 3
    i1 = base + (1 if r > 0 else 0)
    i2 = i1 + base + (1 if r > 1 else 0)
    
    col1_items = input_items[:i1]
    col2_items = input_items[i1:i2]
    col3_items = input_items[i2:]
    
    def build_column_html(items):
        html = "<div style='font-size:16px; line-height:1.8;'>"
        for key, value in items:
            key_formatted = key.replace("_", " ").title()
            html += f"<p><strong>{key_formatted}:</strong> {value}</p>"
        html += "</div>"
        return html
    
    col1_html = build_column_html(col1_items)
    col2_html = build_column_html(col2_items)
    col3_html = build_column_html(col3_items)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(col1_html, unsafe_allow_html=True)
    with col2:
        st.markdown(col2_html, unsafe_allow_html=True)
    with col3:
        st.markdown(col3_html, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------
# Prediction
# -------------------------------------------
if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.subheader(f"Predicted Value: ${prediction:,.2f}")
    st.balloons()

# -------------------------------------------
# Footer
# -------------------------------------------
st.markdown("<div class='footer'>Modern Car Price Predictor App © 2025</div>", unsafe_allow_html=True)

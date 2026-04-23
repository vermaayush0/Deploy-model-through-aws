import os
import pickle

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "weather_forecast_data.csv"
MODEL_PATH = "weather_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
TARGET_COLUMN = "Rain"

st.set_page_config(
    page_title="Smart Agriculture Weather Predictor",
    page_icon="☔",
    layout="centered",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df.dropna()

@st.cache_data
def encode_data(df):
    df = df.copy()
    encoders = {}

    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    return X, y, encoders

def train_model(df):
    X, y, encoders = encode_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoders, f)

    return model, encoders, X_test, y_test

@st.cache_data
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

@st.cache_data
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return accuracy, report

@st.cache_data
def get_prediction(model, input_df):
    return model.predict(input_df)

# Load data and model
raw_data = load_dataset()
model_exists = os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)

if not model_exists:
    model, encoders, X_test, y_test = train_model(raw_data)
    trained_message = "Model trained and ready to predict."
else:
    model, encoders = load_model()
    X, y, _ = encode_data(raw_data)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )
    trained_message = "Loaded existing model."

accuracy, report = evaluate_model(model, X_test, y_test)

st.markdown("# 🌾 Smart Agriculture Weather Predictor")
st.markdown("Predict whether it will rain based on current weather conditions.")

with st.sidebar:
    st.header("Input Parameters")
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    cloud_cover = st.number_input("Cloud Cover (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    pressure = st.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1000.0, step=0.1)
    st.divider()
    st.write("***")
    st.write(trained_message)
    st.write(f"Model accuracy: **{accuracy * 100:.2f}%**")
    st.write("Use the slider inputs above and click the button below.")

with st.container():
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", f"{temperature:.1f} °C")
    col2.metric("Humidity", f"{humidity:.1f} %")
    col3.metric("Wind Speed", f"{wind_speed:.1f} km/h")

    col4, col5 = st.columns(2)
    col4.metric("Cloud Cover", f"{cloud_cover:.1f} %")
    col5.metric("Pressure", f"{pressure:.1f} hPa")

st.markdown("---")

if st.button("Predict Rainfall"):
    input_df = pd.DataFrame(
        [[temperature, humidity, wind_speed, cloud_cover, pressure]],
        columns=["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure"],
    )
    prediction = get_prediction(model, input_df)[0]

    if TARGET_COLUMN in encoders:
        prediction_label = encoders[TARGET_COLUMN].inverse_transform([prediction])[0]
    else:
        prediction_label = "rain" if prediction == 1 else "no rain"

    if prediction_label.lower() in ["rain", "yes", "true", "1"]:
        st.success(f"Prediction: **{prediction_label.title()}** — take precautions! 🌧️")
    else:
        st.info(f"Prediction: **{prediction_label.title()}** — clear skies expected. ☀️")

with st.expander("Model performance and dataset overview", expanded=True):
    st.markdown("### Model summary")
    st.write(f"**Accuracy:** {accuracy * 100:.2f}%")
    st.text(report)
    st.markdown("### Sample of the dataset")
    st.dataframe(raw_data.head())
    st.markdown("### Class distribution")
    st.bar_chart(raw_data[TARGET_COLUMN].value_counts())

st.markdown("---")
st.caption("Built with Streamlit for a simple weather forecast prediction interface.")

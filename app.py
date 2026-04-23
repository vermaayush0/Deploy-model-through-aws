import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Check if model exists, else train
if not os.path.exists("weather_model.pkl"):
    # Load dataset
    df = pd.read_csv("weather_forecast_data.csv")

    # Handle missing values
    df = df.dropna()

    # Convert categorical columns to numeric
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Split features and target
    X = df.drop("Rain", axis=1)
    y = df["Rain"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open("weather_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved successfully!")

st.title("Weather Forecast Prediction")
st.write("Enter the weather parameters to predict if it will rain.")

temperature = st.slider("Temperature", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.slider("Humidity", min_value=0.0, max_value=100.0, value=50.0)
wind_speed = st.slider("Wind Speed", min_value=0.0, max_value=20.0, value=5.0)
cloud_cover = st.slider("Cloud Cover", min_value=0.0, max_value=100.0, value=50.0)
pressure = st.slider("Pressure", min_value=900.0, max_value=1100.0, value=1000.0)

if st.button("Predict"):
    # Load model
    with open("weather_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Prepare input
    input_data = pd.DataFrame([[temperature, humidity, wind_speed, cloud_cover, pressure]], columns=['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure'])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write("It will rain!")
    else:
        st.write("No rain expected.")

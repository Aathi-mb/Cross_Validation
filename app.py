import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="🍫 Chocolate Sales Prediction", layout="centered")
st.title("🍫 Chocolate Sales Prediction App")

DATA_PATH = r"C:\Users\Aathira\Desktop\Validation_cross\Chocolate Sales.csv"
MODEL_PATH = "chocolate_model.pkl"

# ==============================
# Train model once
# ==============================
if not os.path.exists(MODEL_PATH):

    st.info("Training model for first time...")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop(columns=['Date'], inplace=True)

    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    X = df.drop(columns=['Amount'])
    y = df['Amount']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, label_encoders), f)

    st.success("✅ Model trained & saved")

# ==============================
# Load model
# ==============================
with open(MODEL_PATH, "rb") as f:
    model, label_encoders = pickle.load(f)

# ==============================
# Frontend
# ==============================
st.subheader("📌 Enter Sales Details")

sales_person = st.selectbox("Sales Person", label_encoders['Sales Person'].classes_)
country = st.selectbox("Country", label_encoders['Country'].classes_)
product = st.selectbox("Product", label_encoders['Product'].classes_)
boxes = st.number_input("Boxes Shipped", min_value=1, step=1)
date = st.date_input("Date")

# ==============================
# Predict
# ==============================
if st.button("🔮 Predict Sales Amount"):

    input_df = pd.DataFrame({
        'Sales Person': [sales_person],
        'Country': [country],
        'Product': [product],
        'Boxes Shipped': [boxes],
        'Year': [date.year],
        'Month': [date.month],
        'Day': [date.day]
    })

    # ✅ FIXED ENCODING
    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)

    st.success(f"💰 Predicted Sales Amount: **{prediction[0]:.2f}**")

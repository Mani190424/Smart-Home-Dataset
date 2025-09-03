import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import io

# ================================
# Load Data
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("Synthetic_Smart_Home_140k.csv")
    df["Date_Time"] = pd.to_datetime(df["Date_Time"])
    return df

df = load_data()

# ================================
# Train Random Forest Model
# ================================
@st.cache_resource
def train_model(data):
    X = data.drop(columns=["sensor status", "Date_Time", "Room"])
    y = data["sensor status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=300, random_state=42, max_depth=12, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    acc = accuracy_score(y_test, rf.predict(X_test))
    return rf, scaler, acc

rf_model, scaler, model_acc = train_model(df)

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Smart Home Dashboard", layout="wide")

st.title("🏡 Smart Home Dashboard")
st.markdown(f"### 🤖 Random Forest Model Accuracy: **{model_acc*100:.2f}%**")

# Tabs for Rooms
tabs = st.tabs(["🛋️ Living Room", "🛏️ Bedroom", "🍽️ Kitchen", "🌲 Outdoor"])

# Sidebar Filters
st.sidebar.header("🔍 Filters")
room_filter = st.sidebar.multiselect("Select Room(s):", df["Room"].unique(), default=df["Room"].unique())
date_range = st.sidebar.date_input("Select Date Range:", [df["Date_Time"].min(), df["Date_Time"].max()])
time_group = st.sidebar.selectbox("Group by Time:", ["Daily", "Weekly", "Monthly", "Yearly"])

# Apply Filters
filtered = df[(df["Room"].isin(room_filter)) &
              (df["Date_Time"].dt.date >= date_range[0]) &
              (df["Date_Time"].dt.date <= date_range[1])]

# ================================
# KPI Cards
# ================================
def kpi_section(data, room):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Avg Temp", f"{data['Temperature'].mean():.2f} °C")
    col2.metric("💧 Avg Humidity", f"{data['Humidity'].mean():.2f} %")
    col3.metric("⚡ Total Energy", f"{data['Energy_Consumption'].sum():.2f} kWh")
    col4.metric("🌬️ Wind Speed", f"{data['Wind Speed'].mean():.2f} km/h" if room != "Outdoor" else "N/A")

    if room in ["Living Room", "Bedroom", "Kitchen"]:
        st.info(f"💡 Appliances in {room}: Fan & Light available")

# ================================
# Charts
# ================================
def line_charts(data, room):
    st.subheader(f"📊 Sensor Trends - {room}")
    metrics = ["Temperature", "Humidity", "Energy_Consumption"]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8,3))
        data.groupby(data["Date_Time"].dt.date)[metric].mean().plot(ax=ax)
        ax.set_title(f"{metric} Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel(metric)
        st.pyplot(fig)

# ================================
# Download Button
# ================================
def download_button(data):
    buffer = io.BytesIO()
    data.to_excel(buffer, index=False, engine="openpyxl")
    st.download_button(
        label="📥 Download Data (Excel)",
        data=buffer,
        file_name="filtered_smart_home.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ================================
# ML Prediction Section
# ================================
def prediction_section():
    st.subheader("🤖 Predict Sensor Status")
    col1, col2, col3 = st.columns(3)
    temp = col1.number_input("Temperature (°C)", 0, 50, 25)
    hum = col2.number_input("Humidity (%)", 0, 100, 50)
    motion = col3.number_input("Motion", 0, 1, 0)
    light = st.slider("Light Intensity (lux)", 0, 1000, 300)
    energy = st.number_input("Energy Consumption (kWh)", 0.0, 100.0, 10.0)
    wind = st.number_input("Wind Speed (km/h)", 0.0, 50.0, 5.0)

    input_data = pd.DataFrame([[temp, hum, motion, light, energy, wind]],
                              columns=["Temperature","Humidity","Motion","Light Intensity","Energy_Consumption","Wind Speed"])
    input_scaled = scaler.transform(input_data)
    pred = rf_model.predict(input_scaled)[0]

    st.success(f"🔮 Predicted Sensor Status: **{'ON' if pred==1 else 'OFF'}**")

# ================================
# Tabs Content
# ================================
for tab, room in zip(tabs, ["Living Room", "Bedroom", "Kitchen", "Outdoor"]):
    with tab:
        room_data = filtered[filtered["Room"] == room]
        if room_data.empty:
            st.warning(f"No data available for {room}")
        else:
            kpi_section(room_data, room)
            line_charts(room_data, room)
            download_button(room_data)

# Prediction Section
prediction_section()

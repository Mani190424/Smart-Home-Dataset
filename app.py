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
# Load external CSS
# ================================
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
# Sidebar Filters
# ================================
st.sidebar.header("🔍 Filters")
room_filter = st.sidebar.multiselect("Select Room(s):", df["Room"].unique(), default=df["Room"].unique())
date_range = st.sidebar.date_input("Select Date Range:", [df["Date_Time"].min(), df["Date_Time"].max()])
time_group = st.sidebar.selectbox("Group by Time:", ["Daily", "Weekly", "Monthly", "Yearly"])

# Apply Filters
filtered = df[(df["Room"].isin(room_filter)) &
              (df["Date_Time"].dt.date >= date_range[0]) &
              (df["Date_Time"].dt.date <= date_range[1])]

# ================================
# Navbar
# ================================
st.markdown("""
    <div class="navbar">
        <h2>🏡 Smart Home Dashboard</h2>
    </div>
""", unsafe_allow_html=True)

# ================================
# KPI Section
# ================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Avg Temp 🌡️", f"{filtered['Temperature'].mean():.2f} °C")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Avg Humidity 💧", f"{filtered['Humidity'].mean():.2f} %")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Total Energy ⚡", f"{filtered['Energy'].sum():.2f} kWh")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Model Accuracy 🤖", f"{model_acc*100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# ================================
# Line Charts
# ================================
st.subheader("📊 Trends")

fig, ax = plt.subplots()
ax.plot(filtered["Date_Time"], filtered["Temperature"], label="Temperature")
ax.plot(filtered["Date_Time"], filtered["Humidity"], label="Humidity")
ax.plot(filtered["Date_Time"], filtered["Energy"], label="Energy")
ax.legend()
st.pyplot(fig)

# ================================
# Download Button
# ================================
st.subheader("⬇️ Download Data")
buffer = io.BytesIO()
filtered.to_csv(buffer, index=False)
st.download_button("Download CSV", buffer.getvalue(), "filtered_data.csv", "text/csv")

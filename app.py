import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# ================= Login Page =================
def login():
    st.title("🔐 Smart Home Dashboard Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
        else:
            st.error("❌ Invalid credentials")

# ================= Main Dashboard =================
def dashboard(df):
    st.title("🏠 Smart Home Dashboard")

    # Load CSS
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # ===== Date Filter =====
    min_date, max_date = df["Date"].min(), df["Date"].max()
    date_range = st.date_input("📅 Select Date Range", [min_date, max_date])
    if len(date_range) == 2:
        df = df[(df["Date"] >= pd.to_datetime(date_range[0])) &
                (df["Date"] <= pd.to_datetime(date_range[1]))]

    # ===== Room Filter =====
    rooms = st.multiselect("Select Rooms", df["Room"].unique(), default=df["Room"].unique())
    df = df[df["Room"].isin(rooms)]

    # ===== KPI Cards =====
    avg_temp = round(df["Temperature"].mean(), 2)
    avg_hum = round(df["Humidity"].mean(), 2)
    total_energy = round(df["Energy"].sum(), 2)

    st.markdown(f"""
    <div class="kpi-container">
      <div class="kpi-card temp-card">
        <div class="kpi-icon">🌡️</div>
        <div class="kpi-title">Temperature</div>
        <div class="kpi-value">{avg_temp} °C</div>
      </div>

      <div class="kpi-card humidity-card">
        <div class="kpi-icon">💧</div>
        <div class="kpi-title">Humidity</div>
        <div class="kpi-value">{avg_hum} %</div>
      </div>

      <div class="kpi-card energy-card">
        <div class="kpi-icon">⚡</div>
        <div class="kpi-title">Energy</div>
        <div class="kpi-value">{total_energy} kWh</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== Room-wise Comparison =====
    st.subheader("📊 Room-wise Energy Usage")
    room_chart = px.bar(df, x="Room", y="Energy", color="Room", barmode="group")
    st.plotly_chart(room_chart, use_container_width=True)

    # ===== Appliance-wise Comparison =====
    if "Appliance" in df.columns:
        st.subheader("🔌 Appliance-wise Energy Usage")
        app_chart = px.pie(df, names="Appliance", values="Energy", hole=0.4)
        st.plotly_chart(app_chart, use_container_width=True)

# ================= Main =================
def main():
    # Load dataset
    df = pd.read_csv("Smart home dataset.csv")
    # Convert date column
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login()
    else:
        dashboard(df)

if __name__ == "__main__":
    main()

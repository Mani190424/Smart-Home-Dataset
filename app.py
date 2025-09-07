import streamlit as st
import pandas as pd
pip install matplotlib


# ========== LOGIN PAGE ==========
def login():
    st.title("🔐 Smart Home Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
            st.success("✅ Login successful!")
        else:
            st.error("❌ Invalid username or password")

# ========== DASHBOARD PAGE ==========
def dashboard():
    st.title("🏠 Smart Home Dashboard")

    # Load data
    df = pd.read_csv("Smart home dataset.csv")

    # Convert timestamp if available
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Sidebar filters
    st.sidebar.header("🔎 Filters")

    # Date filter
    if "Timestamp" in df.columns:
        min_date, max_date = df["Timestamp"].min(), df["Timestamp"].max()
        start_date, end_date = st.sidebar.date_input("📅 Select Date Range",
                                                     [min_date, max_date],
                                                     min_value=min_date,
                                                     max_value=max_date)
        if isinstance(start_date, list):
            start_date, end_date = start_date[0], start_date[1]
        df = df[(df["Timestamp"].dt.date >= start_date) & (df["Timestamp"].dt.date <= end_date)]

    # Room filter
    if "Room" in df.columns:
        rooms = df["Room"].unique()
        selected_rooms = st.sidebar.multiselect("🏘️ Select Rooms", rooms, default=rooms)
        df = df[df["Room"].isin(selected_rooms)]

    # Appliance filter
    if "Appliance" in df.columns:
        appliances = df["Appliance"].unique()
        selected_appliances = st.sidebar.multiselect("🔌 Select Appliances", appliances, default=appliances)
        df = df[df["Appliance"].isin(selected_appliances)]

    # KPI cards
    st.subheader("📌 Key Metrics")
    col1, col2, col3 = st.columns(3)
    if "Temperature" in df.columns:
        col1.metric("🌡️ Avg Temp", f"{df['Temperature'].mean():.2f} °C")
    if "Humidity" in df.columns:
        col2.metric("💧 Avg Humidity", f"{df['Humidity'].mean():.2f} %")
    if "Energy_Usage" in df.columns:
        col3.metric("⚡ Total Energy", f"{df['Energy_Usage'].sum():.2f} kWh")

    # Room-wise comparison
    if "Room" in df.columns and "Energy_Usage" in df.columns:
        st.subheader("🏘️ Room-wise Energy Usage")
        room_energy = df.groupby("Room")["Energy_Usage"].sum()
        fig, ax = plt.subplots()
        room_energy.plot(kind="bar", ax=ax, color="skyblue")
        ax.set_ylabel("Energy (kWh)")
        st.pyplot(fig)

    # Appliance-wise usage
    if "Appliance" in df.columns and "Energy_Usage" in df.columns:
        st.subheader("🔌 Appliance-wise Energy Usage")
        app_energy = df.groupby("Appliance")["Energy_Usage"].sum()
        fig2, ax2 = plt.subplots()
        app_energy.plot(kind="pie", autopct="%1.1f%%", ax=ax2)
        ax2.set_ylabel("")
        st.pyplot(fig2)

    # Time series - Temperature
    if "Temperature" in df.columns and "Timestamp" in df.columns:
        st.subheader("🌡️ Temperature Trend")
        fig3, ax3 = plt.subplots()
        ax3.plot(df["Timestamp"], df["Temperature"], color="orange")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Temp (°C)")
        plt.xticks(rotation=45)
        st.pyplot(fig3)

    # Time series - Energy Usage
    if "Energy_Usage" in df.columns and "Timestamp" in df.columns:
        st.subheader("⚡ Energy Usage Over Time")
        fig4, ax4 = plt.subplots()
        ax4.plot(df["Timestamp"], df["Energy_Usage"], color="blue")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Energy (kWh)")
        plt.xticks(rotation=45)
        st.pyplot(fig4)

    if st.button("Logout"):
        st.session_state["logged_in"] = False

# ========== MAIN ==========
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    dashboard()

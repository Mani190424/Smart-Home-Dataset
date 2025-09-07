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

        login()
    else:
        dashboard(df)

# Load dataset
df = pd.read_csv("processed_with_ac_timestamp(Sheet1).csv")

# Convert datetime
df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🏠 Smart Home Dashboard")

# --- Room Selector as Cards ---
rooms = ["Living Room", "Bedroom", "Kitchen", "Outdoor"]

cols = st.columns(len(rooms))
selected_room = None

for i, room in enumerate(rooms):
    if cols[i].button(room, key=room):
        selected_room = room

# Default room if nothing selected
if not selected_room:
    selected_room = "Living Room"

st.markdown(f"### {selected_room} Dashboard")

# Filter dataset by room
room_data = df[df["Room"] == selected_room]

# --- KPI Cards ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Avg Temp (°C)", f"{room_data['Temperature'].mean():.2f}")

with kpi2:
    st.metric("Max Humidity (%)", f"{room_data['Humidity'].max():.2f}")

with kpi3:
    st.metric("Min Humidity (%)", f"{room_data['Humidity'].min():.2f}")

with kpi4:
    st.metric("Total Energy (kWh)", f"{room_data['Energy_Usage'].sum():.2f}")

# --- Chart Example ---
st.markdown("#### Temperature Trend")
fig, ax = plt.subplots()
ax.plot(room_data["Date_Time"], room_data["Temperature"], label="Temperature")
ax.set_xlabel("Time")
ax.set_ylabel("Temperature (°C)")
ax.legend()
st.pyplot(fig)

st.markdown("#### Humidity Trend")
fig2, ax2 = plt.subplots()
ax2.plot(room_data["Date_Time"], room_data["Humidity"], color="orange", label="Humidity")
ax2.set_xlabel("Time")
ax2.set_ylabel("Humidity (%)")
ax2.legend()
st.pyplot(fig2)

st.markdown("#### Energy Usage Trend")
fig3, ax3 = plt.subplots()
ax3.plot(room_data["Date_Time"], room_data["Energy_Usage"], color="green", label="Energy")
ax3.set_xlabel("Time")
ax3.set_ylabel("Energy (kWh)")
ax3.legend()
st.pyplot(fig3)

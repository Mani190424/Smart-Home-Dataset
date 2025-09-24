import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------
# Page Config
# -------------------
st.set_page_config(
    page_title="Smart Home Dashboard",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Smart Home Dashboard")

# -------------------
# Load CSV
# -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Smart home dataset.csv")
    # Convert Date_Time to datetime
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce')
    return df

df = load_data()

# Show data sample
with st.expander("📂 View Dataset"):
    st.write(df.head())
    st.write("Columns:", df.columns.tolist())

# -------------------
# Sidebar Filters
# -------------------
st.sidebar.header("🔎 Filters")

# Room filter
rooms = df['Room'].dropna().unique().tolist()
selected_room = st.sidebar.selectbox("Select Room", ["All"] + rooms)

# Date filter
min_date = df['Date_Time'].min()
max_date = df['Date_Time'].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# Filter Data
filtered_df = df.copy()
if selected_room != "All":
    filtered_df = filtered_df[filtered_df['Room'] == selected_room]

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df['Date_Time'] >= pd.to_datetime(start_date)) &
        (filtered_df['Date_Time'] <= pd.to_datetime(end_date))
    ]

# -------------------
# KPIs
# -------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🌡️ Avg Temperature", f"{filtered_df['Temperature'].mean():.2f} °C")
with col2:
    st.metric("💧 Avg Humidity", f"{filtered_df['Humidity'].mean():.2f} %")
with col3:
    st.metric("⚡ Total Energy Usage", f"{filtered_df['Energy_Usage'].sum():.2f} kWh")

# -------------------
# Charts
# -------------------

# Temperature over time
fig_temp = px.line(filtered_df, x="Date_Time", y="Temperature", color="Room", title="🌡️ Temperature Over Time")
st.plotly_chart(fig_temp, use_container_width=True)

# Humidity over time
fig_hum = px.line(filtered_df, x="Date_Time", y="Humidity", color="Room", title="💧 Humidity Over Time")
st.plotly_chart(fig_hum, use_container_width=True)

# Energy Usage
fig_energy = px.bar(filtered_df, x="Date_Time", y="Energy_Usage", color="Room", title="⚡ Energy Usage Over Time")
st.plotly_chart(fig_energy, use_container_width=True)

# Light Intensity (if available)
if "Light_Intensity" in filtered_df.columns:
    fig_light = px.line(filtered_df, x="Date_Time", y="Light_Intensity", color="Room", title="💡 Light Intensity")
    st.plotly_chart(fig_light, use_container_width=True)

# -------------------
# Download Section
# -------------------
st.subheader("⬇️ Download Filtered Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download as CSV",
    data=csv,
    file_name="filtered_smart_home.csv",
    mime="text/csv"
)


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import io

# ================================
# Load Data
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("Smart home dataset.csv")
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")
    return df

df = load_data()

# ================================
# Train Random Forest Model
# ================================
@st.cache_resource
def train_model(data):
    # Drop non-numeric columns except target
    X = data.drop(columns=["sensor status", "Date_Time", "Room", "Appliance_Status"])
    y = data["sensor status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10, n_jobs=-1)
    rf.fit(X_train, y_train)

    acc = accuracy_score(y_test, rf.predict(X_test))
    return rf, scaler, acc

rf_model, scaler, model_acc = train_model(df)

# ================================
# Sidebar Filters
# ================================
st.sidebar.header("🔍 Filters")
room_filter = st.sidebar.multiselect("Select Room(s):", df["Room"].unique(), default=df["Room"].unique())
date_range = st.sidebar.date_input("Select Date Range:", [df["Date_Time"].min().date(), df["Date_Time"].max().date()])
time_group = st.sidebar.selectbox("Group by Time:", ["Daily", "Weekly", "Monthly", "Yearly"])

# Apply Filters
filtered = df[(df["Room"].isin(room_filter)) &
              (df["Date_Time"].dt.date >= date_range[0]) &
              (df["Date_Time"].dt.date <= date_range[1])]

# ================================
# Navbar
# ================================
st.markdown("""
    <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;">
        <h2>🏡 Smart Home Dashboard</h2>
    </div>
""", unsafe_allow_html=True)

# ================================
# KPI Section
# ================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Avg Temp 🌡️", f"{filtered['Temperature'].mean():.2f} °C" if not filtered.empty else "N/A")

with col2:
    st.metric("Avg Humidity 💧", f"{filtered['Humidity'].mean():.2f} %" if not filtered.empty else "N/A")

with col3:
    st.metric("Total Energy ⚡", f"{filtered['Energy_Usage'].sum():.2f} kWh" if not filtered.empty else "N/A")

with col4:
    st.metric("Model Accuracy 🤖", f"{model_acc*100:.2f}%")

# ================================
# Line Charts
# ================================
st.subheader("📊 Trends")

if filtered.empty:
    st.warning("No data available for selected filters!")
else:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(filtered["Date_Time"], filtered["Temperature"], label="Temperature")
    ax.plot(filtered["Date_Time"], filtered["Humidity"], label="Humidity")
    ax.plot(filtered["Date_Time"], filtered["Energy_Usage"], label="Energy Usage")
    ax.set_xlabel("Date Time")
    ax.set_ylabel("Values")
    ax.legend()
    st.pyplot(fig)

    st.line_chart(filtered.set_index("Date_Time")[["Temperature", "Humidity", "Energy_Usage"]])

# ================================
# Download Button
# ================================
st.subheader("⬇️ Download Data")
buffer = io.BytesIO()
filtered.to_csv(buffer, index=False)
st.download_button("Download CSV", buffer.getvalue(), "filtered_data.csv", "text/csv")

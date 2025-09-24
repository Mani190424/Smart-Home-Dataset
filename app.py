# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path
import io

# ------------------ CSS Loader ------------------
def load_css(file_path="style.css"):
    p = Path(file_path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------ Login ------------------
def login():
    st.title("🔐 Smart Home Dashboard - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "Smart" and password == "1234":
            st.session_state["logged_in"] = True
            st.success("Login successful ✅")
            st.stop()
        else:
            st.error("Invalid Username or Password ❌")

# ------------------ Dashboard ------------------
def dashboard():
    load_css("style.css")

    # ---------------- Dashboard Heading ----------------
    st.markdown("""
        <div style="background-color:#4CAF50;padding:15px;border-radius:10px;text-align:center;color:white;">
            <h1>🏡 Smart Home Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)

    # Load CSV
    df = pd.read_csv("Smart home dataset.csv")
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")

    # ---------------- Sidebar Filters ----------------
    st.sidebar.header("⚙️ Filters")
    min_date = df["Date_Time"].min().date()
    max_date = df["Date_Time"].max().date()

    date_range = st.sidebar.date_input(
        "📅 Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    if isinstance(date_range, list) and len(date_range)==2:
        start_date, end_date = date_range
        df = df[(df["Date_Time"].dt.date >= start_date) & (df["Date_Time"].dt.date <= end_date)]

    grouping = st.sidebar.selectbox("⏳ Group Data By", ["Daily","Weekly","Monthly","Yearly"])
    if grouping == "Daily":
        df["Period"] = df["Date_Time"].dt.date
    elif grouping == "Weekly":
        df["Period"] = df["Date_Time"].dt.to_period("W").apply(lambda r: r.start_time)
    elif grouping == "Monthly":
        df["Period"] = df["Date_Time"].dt.to_period("M").apply(lambda r: r.start_time)
    else:
        df["Period"] = df["Date_Time"].dt.to_period("Y").apply(lambda r: r.start_time)

    # ---------------- Room Selection Cards ----------------
    st.markdown("### 🏠 Select Room")
    rooms = df["Room"].dropna().unique()
    selected_room = None
    room_icons = {"Living Room":"🛋️","Bedroom":"🛏️","Kitchen":"🍽️","Outdoor":"🌲"}

    cols = st.columns(len(rooms))
    for i, room in enumerate(rooms):
        if cols[i].button(f"{room_icons.get(room,'🏠')} {room}", key=room):
            selected_room = room
    if not selected_room:
        selected_room = rooms[0]

    st.markdown(f"## {selected_room} ({grouping})")

    room_df = df[df["Room"]==selected_room]

    # ---------------- KPI Cards ----------------
    if not room_df.empty:
        kpi_cols = st.columns(3)
        kpis = [
            ("🌡 Avg Temp", f"{room_df['Temperature'].mean():.2f} °C"),
            ("💧 Avg Humidity", f"{room_df['Humidity'].mean():.2f} %"),
            ("⚡ Total Energy", f"{room_df['Energy_Usage'].sum():.2f} kWh")
        ]
        for col, (title, value) in zip(kpi_cols, kpis):
            col.markdown(f'''
                <div class="kpi-card">{title}<br><b>{value}</b></div>
            ''', unsafe_allow_html=True)

        # ---------------- Trend Charts ----------------
        grouped = room_df.groupby("Period").agg({
            "Temperature":"mean","Humidity":"mean","Energy_Usage":"sum"
        }).reset_index()
        st.markdown("### 📊 Trends")
        st.plotly_chart(px.line(grouped, x="Period", y="Temperature", title="🌡️ Temperature Trend", color_discrete_sequence=["red"]), use_container_width=True)
        st.plotly_chart(px.line(grouped, x="Period", y="Humidity", title="💧 Humidity Trend", color_discrete_sequence=["skyblue"]), use_container_width=True)
        st.plotly_chart(px.line(grouped, x="Period", y="Energy_Usage", title="⚡ Energy Usage Trend", color_discrete_sequence=["green"]), use_container_width=True)

        # ---------------- Room-wise Comparison ----------------
        st.markdown("### 🏘️ Room-wise Comparison")
        selected_rooms = st.multiselect("Select Rooms", rooms, default=rooms[:2])
        compare_df = df[df["Room"].isin(selected_rooms)].groupby("Room").agg({
            "Temperature":"mean","Humidity":"mean","Energy_Usage":"sum"
        }).reset_index()
        st.plotly_chart(px.bar(compare_df, x="Room", y="Energy_Usage", color="Room", barmode="group"), use_container_width=True)

        # ---------------- Appliance Usage ----------------
        st.markdown(f"### 🔌 Appliance Usage in {selected_room}")
        if "Appliance_Status" in room_df.columns:
            app_usage = room_df.groupby("Appliance_Status").size().reset_index(name="Count")
            if not app_usage.empty:
                st.plotly_chart(px.bar(app_usage, x="Appliance_Status", y="Count", color="Appliance_Status"), use_container_width=True)
            else:
                st.info(f"No appliance data for {selected_room}")
        else:
            st.info("Appliance_Status column not found.")

        # ---------------- Correlation Heatmap ----------------
        st.markdown("### 🔗 Correlation Heatmap")
        numeric_cols = ["Temperature","Humidity","Energy_Usage"]
        if all(col in room_df.columns for col in numeric_cols):
            corr = room_df[numeric_cols].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # ---------------- Clustering (2D & 3D) ----------------
        st.markdown("### 🔹 Clustering")
        X = room_df[numeric_cols].dropna()
        if len(X) >= 3:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=3, random_state=42)
            room_df['Cluster'] = kmeans.fit_predict(X_scaled)

            # 2D Scatter
            fig2d = px.scatter(room_df, x="Temperature", y="Humidity", color="Cluster", title="2D Clustering")
            st.plotly_chart(fig2d, use_container_width=True)

            # 3D Scatter
            fig3d = px.scatter_3d(room_df, x="Temperature", y="Humidity", z="Energy_Usage", color="Cluster", title="3D Clustering")
            st.plotly_chart(fig3d, use_container_width=True)

        # ---------------- Download CSV ----------------
        st.markdown("### 📥 Download Filtered Data")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "filtered_smart_home.csv", "text/csv")

        # ---------------- Logout ----------------
        if st.button("Logout"):
            st.session_state["logged_in"]=False
            st.stop()

# ------------------ Main ------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"]=False

if not st.session_state["logged_in"]:
    login()
else:
    dashboard()

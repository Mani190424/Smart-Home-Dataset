import streamlit as st # for web app
import pandas as pd # for data handling
import plotly.express as px # for charts
import singin
from pathlib import Path # for CSS
from sklearn.preprocessing import StandardScaler # for scaling
from sklearn.cluster import KMeans # for clustering
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
        if username == "Mani" and password == "1901":
            st.session_state["logged_in"] = True
            st.success("Login successful ✅")
            st.stop()
        else:
            st.error("Invalid Username or Password ❌")

# ------------------ Dashboard ------------------
def dashboard():
    load_css("style.css")

    # Dashboard Heading
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
    date_range = st.sidebar.date_input("📅 Select Date Range",
                                       [min_date, max_date],
                                       min_value=min_date, max_value=max_date)
    if isinstance(date_range, list) and len(date_range) == 2:
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

    # ---------------- Room Selection ----------------
    st.markdown("### 🏠 Select Room")
    rooms = df["Room"].dropna().unique()
    room_icons = {"Living Room":"🛋️","Bedroom":"🛏️","Kitchen":"🍽️","Outdoor":"🌲"}
    cols = st.columns(len(rooms))
    selected_room = None
    for i, room in enumerate(rooms):
        if cols[i].button(f"{room_icons.get(room,'🏠')} {room}", key=room):
            selected_room = room
    if not selected_room:
        selected_room = rooms[0]

    st.markdown(f"## {selected_room} ({grouping})")
    room_df = df[df["Room"]==selected_room]

    # ---------------- KPI Cards ----------------
    if not room_df.empty:
        col1, col2, col3 = st.columns(3)
        kpi_values = [
            ("🌡 Avg Temp", f"{room_df['Temperature'].mean():.2f} °C"),
            ("💧 Avg Humidity", f"{room_df['Humidity'].mean():.2f} %"),
            ("⚡ Total Energy", f"{room_df['Energy_Usage'].sum():.2f} kWh")
        ]
        for col, (title, value) in zip([col1, col2, col3], kpi_values):
            col.markdown(f'<div class="kpi-card">{title}<br><b>{value}</b></div>', unsafe_allow_html=True)

    # ---------------- Trend Charts ----------------
    st.markdown("### 📊 Trends")
    if not room_df.empty:
        grouped = room_df.groupby("Period").agg({"Temperature":"mean","Humidity":"mean","Energy_Usage":"sum"}).reset_index()
        st.plotly_chart(px.line(grouped, x="Period", y="Temperature", title="🌡️ Temperature Trend", color_discrete_sequence=["red"]), use_container_width=True)
        st.plotly_chart(px.line(grouped, x="Period", y="Humidity", title="💧 Humidity Trend", color_discrete_sequence=["skyblue"]), use_container_width=True)
        st.plotly_chart(px.line(grouped, x="Period", y="Energy_Usage", title="⚡ Energy Usage Trend", color_discrete_sequence=["green"]), use_container_width=True)

    # ---------------- Room-wise Comparison ----------------
    st.markdown("### 🏘️ Room-wise Comparison")
    selected_rooms = st.multiselect("Select Rooms", rooms, default=rooms[:2])
    compare_df = df[df["Room"].isin(selected_rooms)].groupby("Room").agg({"Temperature":"mean","Humidity":"mean","Energy_Usage":"sum"}).reset_index()
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
    numeric_cols = room_df.select_dtypes(include=["float64","int64"]).columns
    if len(numeric_cols) > 1:
        corr = room_df[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                             title=f"Correlation Heatmap - {selected_room}")
        st.plotly_chart(fig_corr, use_container_width=True)

    # ---------------- 2D Clustering ----------------
    st.markdown("### 🧩 2D Clustering (KMeans)")
    if len(numeric_cols) >= 2:
        features = room_df[numeric_cols].dropna()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=3, random_state=42)
        room_df['Cluster_2D'] = kmeans.fit_predict(scaled)
        fig_2d = px.scatter(room_df, x=numeric_cols[0], y=numeric_cols[1],
                            color='Cluster_2D', hover_data=numeric_cols,
                            title=f"2D Clustering - {selected_room}")
        st.plotly_chart(fig_2d, use_container_width=True)

           # ---------------- Confusion Matrix ----------------
    st.markdown("### 🧮 Confusion Matrix (Cluster vs Appliance_Status)")

    if "Appliance_Status" in room_df.columns:
        # Drop NA
        cm_data = room_df.dropna(subset=["Appliance_Status", "Cluster_2D"])

        if not cm_data.empty:
            y_true = cm_data["Appliance_Status"].astype(str)
            y_pred = cm_data["Cluster_2D"].astype(str)

            cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=sorted(y_true.unique()))
            disp.plot(ax=ax, cmap="Blues", colorbar=False)
            st.pyplot(fig)
        else:
            st.info("Not enough data for confusion matrix.")
    else:
        st.info("Appliance_Status column not found for confusion matrix.")

    # ---------------- 3D Clustering ----------------
    st.markdown("### 🧩 3D Clustering (KMeans)")
    if len(numeric_cols) >= 3:
        features_3d = room_df[numeric_cols[:3]].dropna()
        scaler = StandardScaler()
        scaled_3d = scaler.fit_transform(features_3d)
        kmeans_3d = KMeans(n_clusters=3, random_state=42)
        room_df['Cluster_3D'] = kmeans_3d.fit_predict(scaled_3d)
        fig_3d = px.scatter_3d(room_df, x=numeric_cols[0], y=numeric_cols[1], z=numeric_cols[2],
                               color='Cluster_3D', hover_data=numeric_cols,
                               title=f"3D Clustering - {selected_room}")
        st.plotly_chart(fig_3d, use_container_width=True)

    # ---------------- Download CSV ----------------
    st.markdown("### 📥 Download Filtered Data")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "filtered_smart_home.csv", "text/csv")

    # ---------------- Logout ----------------
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.stop()

# ------------------ Main ------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    dashboard()

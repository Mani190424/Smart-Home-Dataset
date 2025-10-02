import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import re
import numpy as np

USER_FILE = os.path.join(os.path.dirname(__file__), "users.csv")

# ------------------ CSS Loader ------------------
def load_css(file_path="style.css"):
    """Load custom CSS styles"""
    try:
        p = Path(file_path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        else:
            # Fallback basic styles
            st.markdown("""
            <style>
            .kpi-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 25px 15px;
                border-radius: 15px;
                color: white;
                text-align: center;
                margin: 5px;
                height: 120px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            }
            .kpi-card b {
                font-size: 1.8rem;
                font-weight: bold;
                display: block;
                margin-top: 8px;
            }
            </style>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading CSS: {e}")

# ------------------ Password Validation ------------------
def validate_password(password):
    """
    Validate password strength:
    - At least 8 characters long
    - Contains at least one letter
    - Contains at least one number
    - Contains at least one special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[a-zA-Z]', password):
        return False, "Password must contain at least one letter"
    
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character (!@#$%^&*(), etc.)"
    
    return True, "Password is strong"

# ------------------ User Functions ------------------
def load_users():
    """Load users from CSV with robust error handling"""
    try:
        if not os.path.exists(USER_FILE):
            df = pd.DataFrame(columns=["username", "password"])
            df.to_csv(USER_FILE, index=False)
            return df
        
        if os.path.getsize(USER_FILE) == 0:
            df = pd.DataFrame(columns=["username", "password"])
            df.to_csv(USER_FILE, index=False)
            return df
        
        users = pd.read_csv(USER_FILE)
        
        if not all(col in users.columns for col in ["username", "password"]):
            df = pd.DataFrame(columns=["username", "password"])
            df.to_csv(USER_FILE, index=False)
            return df
            
        return users
        
    except Exception as e:
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv(USER_FILE, index=False)
        return df

def save_user(username, password):
    """Save new user to CSV"""
    try:
        users = load_users()
        
        if not users.empty and username in users["username"].values:
            st.error(f"Username '{username}' already exists!")
            return False
        
        new_user = pd.DataFrame([{"username": username, "password": password}])
        users = pd.concat([users, new_user], ignore_index=True)
        users.to_csv(USER_FILE, index=False)
        
        st.success(f"User '{username}' saved successfully!")
        return True
        
    except Exception as e:
        st.error(f"Error saving user: {e}")
        return False

def authenticate_user(username, password):
    """Authenticate user"""
    try:
        users = load_users()
        
        if users.empty:
            return False
        
        user_match = users[
            (users["username"].astype(str) == username) & 
            (users["password"].astype(str) == password)
        ]
        
        return len(user_match) > 0
        
    except Exception as e:
        st.error(f"Authentication error: {e}")
        return False

def login():
    """Login/Signup page with parallel buttons"""
    # Load CSS first
    load_css("style.css")
    
    # Main title
    st.markdown("""
        <div style="text-align: center; color: white; margin-bottom: 2rem;">
            <h1>🔐 Smart Home Dashboard</h1>
            <p style="opacity: 0.8;">Secure Login & Analytics Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for the form
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Login")
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("🚀 Login", use_container_width=True):
            if not login_user.strip():
                st.error("Please enter username")
            elif not login_pass.strip():
                st.error("Please enter password")
            else:
                if authenticate_user(login_user.strip(), login_pass.strip()):
                    st.session_state["logged_in"] = True
                    st.session_state["current_user"] = login_user
                    st.success("Login successful! ✅")
                    st.rerun()
                else:
                    st.error("Invalid credentials ❌")
    
    with col2:
        st.markdown("### Sign Up")
        signup_user = st.text_input("Choose Username", key="signup_user")
        signup_pass = st.text_input("Create Password", type="password", key="signup_pass")
        confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")
        
        # Password strength indicator
        if signup_pass:
            is_valid, message = validate_password(signup_pass)
            if is_valid:
                st.success("✅ " + message)
            else:
                st.error("❌ " + message)
        
        if st.button("✨ Create Account", use_container_width=True):
            if not signup_user.strip():
                st.error("Please enter username")
            elif not signup_pass.strip():
                st.error("Please enter password")
            elif signup_pass != confirm_pass:
                st.error("Passwords don't match ❌")
            else:
                is_valid, message = validate_password(signup_pass)
                if not is_valid:
                    st.error(f"Weak password: {message}")
                else:
                    if save_user(signup_user.strip(), signup_pass.strip()):
                        st.success("Account created! Please login. ✅")
                    else:
                        st.error("Username exists ❌")
    
    # Password requirements
    with st.expander("🔒 Password Requirements"):
        st.markdown("""
        - **🔢 At least 8 characters long**
        - **🔤 Contains letters** (a-z, A-Z)
        - **1️⃣ Contains numbers** (0-9)
        - **⚡ Contains special characters** (!@#$%^&*(), etc.)
        
        **Examples:** `SmartHome123!`, `MyHome@2024`
        """)

# ------------------ Dashboard ------------------
def dashboard():
    load_css("style.css")

    # Dashboard Heading
    st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 15px; text-align: center; color: white; 
                    margin-bottom: 2rem; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
            <h1>🏡 Smart Home Dashboard</h1>
            <p style="margin: 0; opacity: 0.9;">Welcome back, {}! 👋</p>
        </div>
    """.format(st.session_state.get("current_user", "User")), unsafe_allow_html=True)

    try:
        # Load CSV
        df = pd.read_csv("Smart home dataset.csv")
        df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")
    except FileNotFoundError:
        st.error("❌ Dataset file 'Smart home dataset.csv' not found.")
        st.info("Please make sure the dataset file is in the same directory as your app.")
        
        # Create sample data for demonstration
        st.warning("📊 Showing sample data for demonstration...")
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
        rooms = ['Living Room', 'Bedroom', 'Kitchen', 'Outdoor']
        
        sample_data = []
        for date in dates:
            for room in rooms:
                sample_data.append({
                    'Date_Time': date,
                    'Room': room,
                    'Temperature': np.random.normal(22, 3),
                    'Humidity': np.random.normal(50, 10),
                    'Energy_Usage': np.random.exponential(2),
                    'Appliance_Status': np.random.choice(['On', 'Off', 'Standby'], p=[0.3, 0.5, 0.2])
                })
        
        df = pd.DataFrame(sample_data)
        st.success("✅ Sample data loaded successfully!")

    # ---------------- Sidebar Filters ----------------
    st.sidebar.header("⚙️ Filters")
    
    # Date range filter
    min_date = df["Date_Time"].min().date()
    max_date = df["Date_Time"].max().date()
    date_range = st.sidebar.date_input(
        "📅 Select Date Range",
        [min_date, max_date],
        min_value=min_date, 
        max_value=max_date
    )
    
    if isinstance(date_range, list) and len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df["Date_Time"].dt.date >= start_date) & (df["Date_Time"].dt.date <= end_date)]

    # Grouping filter
    grouping = st.sidebar.selectbox("⏳ Group Data By", ["Daily", "Weekly", "Monthly", "Yearly"])
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
    room_icons = {"Living Room": "🛋️", "Bedroom": "🛏️", "Kitchen": "🍽️", "Outdoor": "🌲"}
    
    cols = st.columns(len(rooms))
    selected_room = st.session_state.get("selected_room", rooms[0])
    
    for i, room in enumerate(rooms):
        if cols[i].button(f"{room_icons.get(room, '🏠')} {room}", key=room, use_container_width=True):
            selected_room = room
            st.session_state["selected_room"] = room
    
    st.markdown(f"## {room_icons.get(selected_room, '🏠')} {selected_room} ({grouping})")
    room_df = df[df["Room"] == selected_room]

    # ---------------- KPI Cards - FIXED EQUAL SIZE ----------------
    st.markdown("### 📊 Key Performance Indicators")
    
    if not room_df.empty:
        # Create 4 equal columns
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        avg_temp = room_df['Temperature'].mean()
        avg_humidity = room_df['Humidity'].mean()
        total_energy = room_df['Energy_Usage'].sum()
        avg_energy = room_df['Energy_Usage'].mean()
        
        # KPI 1 - Temperature
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div>🌡️ Temperature</div>
                <b>{avg_temp:.1f} °C</b>
            </div>
            """, unsafe_allow_html=True)
        
        # KPI 2 - Humidity
        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div>💧 Humidity</div>
                <b>{avg_humidity:.1f} %</b>
            </div>
            """, unsafe_allow_html=True)
        
        # KPI 3 - Total Energy
        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div>⚡ Total Energy</div>
                <b>{total_energy:.1f} kWh</b>
            </div>
            """, unsafe_allow_html=True)
        
        # KPI 4 - Average Energy
        with col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div>🔌 Avg Energy</div>
                <b>{avg_energy:.1f} kWh</b>
            </div>
            """, unsafe_allow_html=True)

    # ---------------- Trend Charts ----------------
    st.markdown("### 📈 Trends Over Time")
    if not room_df.empty:
        grouped = room_df.groupby("Period").agg({
            "Temperature": "mean",
            "Humidity": "mean", 
            "Energy_Usage": "sum"
        }).reset_index()
        
        # Temperature Trend
        fig_temp = px.line(
            grouped, 
            x="Period", 
            y="Temperature", 
            title="🌡️ Temperature Trend",
            color_discrete_sequence=["#FF6B6B"]
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # Humidity Trend
        fig_humidity = px.line(
            grouped, 
            x="Period", 
            y="Humidity", 
            title="💧 Humidity Trend",
            color_discrete_sequence=["#4ECDC4"]
        )
        st.plotly_chart(fig_humidity, use_container_width=True)
        
        # Energy Usage Trend
        fig_energy = px.line(
            grouped, 
            x="Period", 
            y="Energy_Usage", 
            title="⚡ Energy Usage Trend",
            color_discrete_sequence=["#45B7D1"]
        )
        st.plotly_chart(fig_energy, use_container_width=True)

    # ---------------- Room-wise Comparison ----------------
    st.markdown("### 🏘️ Room-wise Comparison")
    selected_rooms = st.multiselect(
        "Select rooms to compare:", 
        rooms, 
        default=rooms[:min(3, len(rooms))]
    )
    
    if selected_rooms:
        compare_df = df[df["Room"].isin(selected_rooms)].groupby("Room").agg({
            "Temperature": "mean",
            "Humidity": "mean",
            "Energy_Usage": "sum"
        }).reset_index()
        
        # Energy Usage Comparison
        fig_compare = px.bar(
            compare_df, 
            x="Room", 
            y="Energy_Usage", 
            color="Room",
            title="⚡ Energy Usage by Room",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    # ---------------- 2D Clustering ----------------
    st.markdown("### 🧩 2D Clustering (KMeans)")
    
    # Get numeric columns for clustering - FIXED: Use proper dtype selection
    numeric_cols = room_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numeric_cols) >= 2:
        # Select features for clustering
        feature_cols = st.multiselect(
            "Select features for clustering:",
            numeric_cols,
            default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
        )
        
        if len(feature_cols) >= 2:
            # Prepare data for clustering
            features = room_df[feature_cols].dropna()
            
            if len(features) > 0:
                # Scale the features
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                
                # Perform KMeans clustering
                n_clusters = st.slider("Number of clusters:", min_value=2, max_value=6, value=3)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_features)
                
                # Add cluster labels to the features dataframe
                features_with_clusters = features.copy()
                features_with_clusters['Cluster'] = cluster_labels
                
                # Create 2D scatter plot
                fig_cluster = px.scatter(
                    features_with_clusters,
                    x=feature_cols[0],
                    y=feature_cols[1],
                    color='Cluster',
                    title=f"2D Clustering - {selected_room}",
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                # ---------------- Confusion Matrix ----------------
                st.markdown("### 🧮 Confusion Matrix (Cluster vs Appliance Status)")
                
                if "Appliance_Status" in room_df.columns:
                    # Merge cluster labels with original data
                    merged_data = room_df.loc[features.index].copy()
                    merged_data['Cluster'] = cluster_labels
                    
                    # Remove rows with missing appliance status
                    valid_data = merged_data.dropna(subset=['Appliance_Status', 'Cluster'])
                    
                    if not valid_data.empty and len(valid_data['Appliance_Status'].unique()) > 1:
                        # Create confusion matrix
                        y_true = valid_data['Appliance_Status'].astype(str)
                        y_pred = valid_data['Cluster'].astype(str)
                        
                        # Get unique labels
                        true_labels = sorted(y_true.unique())
                        pred_labels = sorted(y_pred.unique())
                        
                        # Create confusion matrix
                        cm = confusion_matrix(y_true, y_pred, labels=true_labels)
                        
                        # Display confusion matrix
                        fig, ax = plt.subplots(figsize=(8, 6))
                        disp = ConfusionMatrixDisplay(
                            confusion_matrix=cm,
                            display_labels=true_labels
                        )
                        disp.plot(ax=ax, cmap='Blues', colorbar=False)
                        ax.set_title(f'Confusion Matrix - {selected_room}')
                        ax.set_xlabel('Predicted Cluster')
                        ax.set_ylabel('Actual Appliance Status')
                        
                        # Rotate x-axis labels if needed
                        plt.xticks(rotation=45)
                        plt.yticks(rotation=45)
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                        
                        # Display some statistics
                        st.markdown("#### 📊 Cluster Statistics")
                        cluster_stats = valid_data.groupby('Cluster')['Appliance_Status'].value_counts().unstack(fill_value=0)
                        st.dataframe(cluster_stats)
                        
                        # Display cluster interpretation
                        st.markdown("#### 🔍 Cluster Interpretation")
                        cluster_means = valid_data.groupby('Cluster')[feature_cols].mean()
                        st.dataframe(cluster_means.style.format("{:.2f}"))
                        
                    else:
                        st.warning("Not enough data with appliance status for confusion matrix. Need at least 2 different appliance status values.")
                else:
                    st.info("Appliance_Status column not found in the dataset.")
            else:
                st.warning("No valid data available for clustering after removing missing values.")
        else:
            st.info("Please select at least 2 features for clustering.")
    else:
        st.info("Not enough numeric columns available for clustering. Need at least 2 numeric columns.")

    # ---------------- 3D Clustering ----------------
    st.markdown("### 🧩 3D Clustering (KMeans)")
    
    if len(numeric_cols) >= 3:
        # Select features for 3D clustering
        feature_cols_3d = st.multiselect(
            "Select 3 features for 3D clustering:",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
            key="3d_features"
        )
        
        if len(feature_cols_3d) == 3:
            # Prepare data for 3D clustering
            features_3d = room_df[feature_cols_3d].dropna()
            
            if len(features_3d) > 0:
                # Scale the features
                scaler_3d = StandardScaler()
                scaled_features_3d = scaler_3d.fit_transform(features_3d)
                
                # Perform KMeans clustering for 3D
                n_clusters_3d = st.slider("Number of clusters (3D):", min_value=2, max_value=6, value=3, key="3d_clusters")
                kmeans_3d = KMeans(n_clusters=n_clusters_3d, random_state=42, n_init=10)
                cluster_labels_3d = kmeans_3d.fit_predict(scaled_features_3d)
                
                # Add cluster labels to the features dataframe
                features_3d_with_clusters = features_3d.copy()
                features_3d_with_clusters['Cluster_3D'] = cluster_labels_3d
                
                # Create 3D scatter plot
                fig_3d = px.scatter_3d(
                    features_3d_with_clusters,
                    x=feature_cols_3d[0],
                    y=feature_cols_3d[1],
                    z=feature_cols_3d[2],
                    color='Cluster_3D',
                    title=f"3D Clustering - {selected_room}",
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("No valid data available for 3D clustering after removing missing values.")
        else:
            st.info("Please select exactly 3 features for 3D clustering.")
    else:
        st.info("Not enough numeric columns available for 3D clustering. Need at least 3 numeric columns.")

    # ---------------- Download CSV ----------------
    st.markdown("### 📥 Download Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='smart_home_data.csv',
        mime='text/csv',
    )

    # ---------------- Logout ----------------
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["current_user"] = None
        st.session_state.pop("selected_room", None)
        st.rerun()

# ------------------ Main ------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    dashboard()

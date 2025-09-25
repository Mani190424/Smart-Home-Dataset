# 🏡 Smart Home Dashboard (IoT + Streamlit)

A Smart Home Dashboard built with **Streamlit**, **Pandas**, **Plotly**, and **Scikit-learn**.  
This project allows you to **analyze IoT sensor data** (temperature, humidity, energy usage, etc.), visualize **room-wise trends**, and perform **clustering** to detect hidden patterns.

---

## ✨ Features
- 🔐 **Login System** (secure access)
- 📅 **Date Range Filter** (daily, weekly, monthly, yearly grouping)
- 🏠 **Room Selection** with icons (Living Room 🛋️, Bedroom 🛏️, Kitchen 🍽️, Outdoor 🌲)
- 📊 **KPI Cards** for Avg Temp, Avg Humidity, Total Energy
- 📈 **Trends Visualization** (Temperature, Humidity, Energy Usage)
- 🏘️ **Room-wise Comparison**
- 🔌 **Appliance Usage Analysis**
- 🔗 **Correlation Heatmap**
- 🧩 **2D & 3D Clustering (KMeans)**
- 🧮 **Confusion Matrix** (Cluster vs Appliance_Status)
- 📥 **Download Filtered Data as CSV**
- 🚪 **Login / Logout System**

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) – Web App
- [Pandas](https://pandas.pydata.org/) – Data Handling
- [Plotly](https://plotly.com/python/) – Interactive Charts
- [Scikit-learn](https://scikit-learn.org/) – Clustering & Confusion Matrix
- [Matplotlib](https://matplotlib.org/) – Confusion Matrix Visualization

---

## 📂 Project Structure
```
📁 Smart-Home-Dashboard
│── app.py                # Main Streamlit app
│── Smart home dataset.csv # Dataset
│── style.css             # Custom CSS styling
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/Smart-Home-Dashboard.git
   cd Smart-Home-Dashboard
   ```

2. Create & activate virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 🔑 Login Details
- **Username:** `Mani`  
- **Password:** `1901`  

(You can change this in `app.py` under the `login()` function.)

---

## 📸 Screenshots
<img width="1920" height="1008" alt="image" src="https://github.com/user-attachments/assets/8464d156-bc78-4b48-b47c-7b13153e18a8" />


---

## 🚀 Deployment
You can deploy this app easily on **[Streamlit Cloud](https://share.streamlit.io/)**.  
Just push your repo to GitHub and connect it to Streamlit Cloud.

---

## 👨‍💻 Author
- Developed by **Manikandan B**  
- Inter Project

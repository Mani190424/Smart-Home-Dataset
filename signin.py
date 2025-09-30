import streamlit as st
import pandas as pd
import os

USER_FILE = "users.csv"

# Load users
def load_users():
    if os.path.exists(USER_FILE):
        return pd.read_csv(USER_FILE)
    else:
        return pd.DataFrame(columns=["username", "password"])

# Save new user
def save_user(username, password):
    users = load_users()
    if username in users["username"].values:
        return False
    new_user = pd.DataFrame([[username, password]], columns=["username", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_FILE, index=False)
    return True

def signin_page():
    st.title("📝 Sign Up - Smart Home Dashboard")

    username = st.text_input("Enter Username")
    password = st.text_input("Enter Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if password != confirm:
            st.error("❌ Passwords do not match!")
        elif username == "" or password == "":
            st.error("⚠️ All fields required!")
        else:
            if save_user(username, password):
                st.success("✅ Account created! Go to Login page.")
                st.page_link("app.py", label="➡️ Go to Login")
            else:
                st.error("⚠️ Username already exists!")

if __name__ == "__main__":
    signin_page()

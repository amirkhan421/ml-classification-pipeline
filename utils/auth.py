# /utils/auth.py
import streamlit as st
import hashlib

# Simple user database (in production, use proper database)
USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "amir": hashlib.sha256("amir123".encode()).hexdigest(),
}

def check_auth():
    """Check if user is authenticated"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        with st.sidebar:
            st.title("🔐 Login")
            username = st.text_input("Username", key="auth_user")
            password = st.text_input("Password", type="password", key="auth_pass")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", use_container_width=True):
                    if username in USERS:
                        hashed = hashlib.sha256(password.encode()).hexdigest()
                        if USERS[username] == hashed:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.rerun()
                        else:
                            st.error("Wrong password!")
                    else:
                        st.error("User not found!")
            
            with col2:
                if st.button("Guest Login", use_container_width=True):
                    st.session_state.logged_in = True
                    st.session_state.username = "Guest"
                    st.rerun()
        
        st.stop()

def logout():
    """Logout user"""
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()
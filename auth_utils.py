import hashlib
import streamlit as st
from database import get_user, create_user

def hash_password(password: str) -> str:
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(password: str, hashed: str) -> bool:
    """Verifies a password against a hash."""
    return hash_password(password) == hashed

def login_user(username, password):
    user = get_user(username)
    if user and check_password(password, user['password_hash']):
        st.session_state.user_id = user['id']
        st.session_state.user_name = user['name']
        st.session_state.username = user['username']
        return True
    return False

def register_user(name, username, password):
    hashed = hash_password(password)
    return create_user(name, username, hashed)

def logout_user():
    for key in ['user_id', 'user_name', 'username']:
        if key in st.session_state:
            del st.session_state[key]

import streamlit as st
import os
import sys
import importlib
from database import Base, engine


# Create database tables if they don't exist
Base.metadata.create_all(bind=engine)

# This must be the first Streamlit command
st.set_page_config(
    page_title="Resume Matcher",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS
st.markdown("""
<style>
    .sidebar-content {
        padding-top: 20px;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        background-color: #f8f9fa;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .avatar-container {
        text-align: center;
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Create directory for session data if it doesn't exist
SESSION_DIR = "./.streamlit_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

# Load sessions to check authentication status
import pickle

student_authenticated = False
admin_authenticated = False

# Check student session
try:
    student_session_file = os.path.join(SESSION_DIR, "student_session.pkl")
    if os.path.exists(student_session_file):
        with open(student_session_file, "rb") as f:
            student_session_data = pickle.load(f)
            if "student_authenticated" in student_session_data and student_session_data["student_authenticated"]:
                student_authenticated = True
except Exception:
    pass

# Check admin session
try:
    admin_session_file = os.path.join(SESSION_DIR, "admin_session.pkl")
    if os.path.exists(admin_session_file):
        with open(admin_session_file, "rb") as f:
            admin_session_data = pickle.load(f)
            if "admin_authenticated" in admin_session_data and admin_session_data["admin_authenticated"]:
                admin_authenticated = True
except Exception:
    pass

# Set environment variable
os.environ["STREAMLIT_IMPORTED_BY_MAIN"] = "true"

# If already authenticated in any portal, go directly to that portal
if student_authenticated:
    app_mode = "Student Portal"
elif admin_authenticated:
    app_mode = "Admin Portal"
else:
    # Application selection sidebar (only shown if not logged in)
    with st.sidebar:
        st.title("Resume Matcher")
        st.image("https://img.icons8.com/fluency/96/000000/resume.png", width=100)
        
        app_mode = st.selectbox(
            "Choose Portal", 
            ["Student Portal", "Admin Portal"]
        )

# Run the appropriate portal code
if app_mode == "Student Portal":
    import student_portal
    
    # Call the function if it exists, otherwise execute main student portal code
    if hasattr(student_portal, 'run_student_portal'):
        student_portal.run_student_portal()
    
elif app_mode == "Admin Portal":
    import admin_portal
    
    if hasattr(admin_portal, 'run_admin_portal'):
        admin_portal.run_admin_portal()

# Footer
with st.sidebar:
    st.markdown("---")
    st.markdown("© 2023 Resume Matcher | v1.0.0")
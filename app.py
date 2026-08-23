import streamlit as st

st.set_page_config(
    page_title="ResumeIQ | AI Candidate OS",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

import subprocess
import sys

# Ensure Playwright library and Chromium binary are installed on startup (cached to run only once)
@st.cache_resource
def ensure_playwright_installed():
    try:
        import playwright
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
    
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            p.chromium.launch(headless=True)
    except Exception:
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)

ensure_playwright_installed()

from auth_utils import login_user, register_user, logout_user




# Hide Streamlit elements and apply light mode css
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    [data-testid="stAppDeployButton"] {display: none;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Session State initialization
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'theme_accent' not in st.session_state:
    st.session_state.theme_accent = "Green"

from ui_utils import inject_custom_css
inject_custom_css()

# --- Auth Gateway ---
if st.session_state.user_id is None:
    # Hide sidebar completely on login page to prevent confusion
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none !important;}
        [data-testid="collapsedControl"] {display: none !important;}
        
        /* Fix mobile distortion by hiding empty padding columns */
        @media (max-width: 768px) {
            [data-testid="column"]:nth-of-type(1),
            [data-testid="column"]:nth-of-type(3) {
                display: none !important;
            }
            [data-testid="column"]:nth-of-type(2) {
                width: 100% !important;
                min-width: 100% !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Use responsive clamp() for fonts so mobile view doesn't break
    st.markdown("<h1 class='gradient-title' style='text-align: center; font-size: clamp(2rem, 6vw, 3.5rem); margin-bottom: 5px; padding-bottom: 10px;'>ResumeIQ Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: clamp(0.9rem, 3vw, 1.2rem); color: #94A3B8;'>Welcome to the world's smartest AI Candidate OS.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        tab1, tab2 = st.tabs([" Login", " Sign Up"])
        
        with tab1:
            st.subheader("Login")
            l_user = st.text_input("Username", key="l_user")
            l_pass = st.text_input("Password", type="password", key="l_pass")
            if st.button("Login", type="primary", use_container_width=True):
                if login_user(l_user, l_pass):
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
                    
        with tab2:
            st.subheader("Sign Up")
            r_name = st.text_input("Full Name", key="r_name")
            r_user = st.text_input("Username", key="r_user")
            r_pass = st.text_input("Password", type="password", key="r_pass")
            if st.button("Sign Up", type="primary", use_container_width=True):
                if register_user(r_name, r_user, r_pass):
                    st.success("Account created successfully! Please login.")
                else:
                    st.error("Username already exists.")
                    
else:
    # Get active colors to render logo block matching the accent
    accent = st.session_state.get("theme_accent", "Green")
    accent_colors = {
        "Blue": "#60A5FA", "Green": "#1ed760", "Red": "#fb7185", "Purple": "#a78bfa", "Amber": "#facc15"
    }
    glow_colors = {
        "Blue": "rgba(96, 165, 250, 0.3)", "Green": "rgba(30, 215, 96, 0.3)", 
        "Red": "rgba(251, 113, 133, 0.3)", "Purple": "rgba(167, 139, 250, 0.3)", "Amber": "rgba(250, 204, 21, 0.3)"
    }
       
    # 1. Hide Sidebar and Native Header to reclaim the top space!
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {display: none !important;}
            [data-testid="collapsedControl"] {display: none !important;}
            [data-testid="stHeader"] {display: none !important;}
            .block-container,
            [data-testid="block-container"],
            [data-testid="stAppViewBlockContainer"],
            [data-testid="stMainBlockContainer"] {
                padding: 1rem 2rem 5rem 2rem !important;
                max-width: 100% !important;
            }
            /* Make nav links look like a sleek top menu */
            .stPageLink { text-align: center !important; }
            .stPageLink a { padding: 8px 12px !important; border-radius: 8px !important; transition: all 0.2s; }
            .stPageLink a:hover { background: rgba(255,255,255,0.05) !important; }
        </style>
    """, unsafe_allow_html=True)
    
    pages = [
        st.Page("pages/1__Dashboard.py", title="Dashboard"),
        st.Page("pages/2__Resume_Analysis.py", title="Resume Analysis"),
        st.Page("pages/3__Resume_Editor.py", title="Resume Editor"),
        st.Page("pages/3__Job_Matches.py", title="Job Matches"),
        st.Page("pages/5__Interview_Prep.py", title="Interview Prep"),
        st.Page("pages/6__Analytics.py", title="Analytics"),
        st.Page("pages/7__Leaderboard.py", title="Leaderboard"),
        st.Page("pages/5__Jobscan_Matcher.py", title="Jobscan Matcher")
    ]
    
    # 2. Build the Top Navigation Bar
    nav_col1, nav_col2, nav_col3 = st.columns([2.2, 8.3, 1.5])
    
    with nav_col1:
        st.markdown(f"""
        <div style='display: flex; align-items: center; gap: 12px; margin-top: 0px; margin-bottom: 10px;'>
            <div style='background-color: {accent_colors[accent]}; width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 12px {glow_colors[accent]};'>
                <span style='color: #000000; font-size: 15px; font-weight: 900; font-family: "Outfit", sans-serif; letter-spacing: -0.5px;'>IQ</span>
            </div>
            <div>
                <h3 style='margin: 0; font-size: 18px; font-family: "Outfit", sans-serif; color: #F4F4F5; font-weight: 700;'>ResumeIQ</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with nav_col2:
        # Horizontal layout for the main pages
        st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
        with c1: st.page_link(pages[0], label="Home")
        with c2: st.page_link(pages[1], label="Analyze")
        with c3: st.page_link(pages[2], label="Editor")
        with c4: st.page_link(pages[3], label="Jobs")
        with c5: st.page_link(pages[4], label="Prep")
        with c6: st.page_link(pages[5], label="Stats")
        with c7: st.page_link(pages[6], label="Rank")
        with c8: st.page_link(pages[7], label="Jobscan")

    with nav_col3:
        st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)
        tc1, tc2 = st.columns(2)
        with tc1:
            with st.popover("", help="Change Theme", use_container_width=True):
                st.markdown("**Select Accent**")
                if st.button(" Blue", use_container_width=True):
                    st.session_state.theme_accent = "Blue"
                    st.rerun()
                if st.button(" Green", use_container_width=True):
                    st.session_state.theme_accent = "Green"
                    st.rerun()
                if st.button(" Red", use_container_width=True):
                    st.session_state.theme_accent = "Red"
                    st.rerun()
                if st.button(" Purple", use_container_width=True):
                    st.session_state.theme_accent = "Purple"
                    st.rerun()
                if st.button(" Amber", use_container_width=True):
                    st.session_state.theme_accent = "Amber"
                    st.rerun()
        with tc2:
            if st.button("", help="Logout", use_container_width=True):
                logout_user()
                st.rerun()
                
    st.markdown("<hr style='margin-top: 0px; margin-bottom: 24px; border: none; height: 1px; background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0) 100%);'>", unsafe_allow_html=True)
    
    inject_custom_css()
    
    # Pages are already defined above

    # Run hidden native navigation
    pg = st.navigation(pages, position="hidden")
    pg.run()

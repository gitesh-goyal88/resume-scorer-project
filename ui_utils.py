import streamlit as st

def inject_custom_css():
    # Set default accent if not in state
    if "theme_accent" not in st.session_state:
        st.session_state.theme_accent = "Green"
        
    accent = st.session_state.theme_accent
    
    # Define high-contrast clean accents matching the screenshot (bright accent + black button text)
    color_map = {
        "Blue": {
            "primary": "#60A5FA", 
            "hover": "#93C5FD", 
            "gradient": "linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%)", 
            "bg_trans": "rgba(96, 165, 250, 0.15)", 
            "border": "rgba(96, 165, 250, 0.4)"
        },
        "Green": {
            "primary": "#22C55E", 
            "hover": "#16a34a", 
            "gradient": "linear-gradient(135deg, #22C55E 0%, #16a34a 100%)", 
            "bg_trans": "rgba(34, 197, 94, 0.15)", 
            "border": "rgba(34, 197, 94, 0.4)"
        },
        "Red": {
            "primary": "#fb7185", 
            "hover": "#fda4af", 
            "gradient": "linear-gradient(135deg, #fb7185 0%, #f43f5e 100%)", 
            "bg_trans": "rgba(251, 113, 133, 0.15)", 
            "border": "rgba(251, 113, 133, 0.4)"
        },
        "Purple": {
            "primary": "#a78bfa", 
            "hover": "#c084fc", 
            "gradient": "linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%)", 
            "bg_trans": "rgba(167, 139, 250, 0.15)", 
            "border": "rgba(167, 139, 250, 0.4)"
        },
        "Amber": {
            "primary": "#facc15", 
            "hover": "#fde047", 
            "gradient": "linear-gradient(135deg, #facc15 0%, #eab308 100%)", 
            "bg_trans": "rgba(250, 204, 21, 0.15)", 
            "border": "rgba(250, 204, 21, 0.4)"
        }
    }
    
    colors = color_map.get(accent, color_map["Green"])
    
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Target only markdown and content tags for typography to keep input frames clean */
    .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown div, .stMarkdown label,
    [data-testid="stSidebar"] p, [data-testid="stHeader"] {{
        font-family: 'Inter', sans-serif !important;
        line-height: 1.7 !important;
        font-size: 1.05rem !important;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #FAFAFA !important;
    }}

    /* Canvas styling - Pitch Dark with Dynamic Ambient Glow */
    .stApp {{
        background-color: #09090B !important;
        background-image: radial-gradient(ellipse at 50% -20%, {colors['bg_trans']} 0%, rgba(9, 9, 11, 1) 70%) !important;
        background-attachment: fixed !important;
        color: #FAFAFA !important;
    }}
    
    [data-testid="stHeader"] {{
        background-color: #09090B !important;
    }}
    
    /* Sidebar matching the screenshot styling with increased width */
    [data-testid="stSidebar"] {{
        background-color: #111216 !important; /* Surface background */
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        min-width: 360px !important;
        max-width: 360px !important;
    }}
    
    /* Sidebar Navigation Links - Scaled Up */
    div[data-testid="stSidebarNavItems"] ul li a {{
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        margin: 6px 12px !important;
        font-size: 15px !important;
        transition: all 0.2s ease !important;
    }}
    div[data-testid="stSidebarNavItems"] ul li a:hover {{
        background-color: rgba(255, 255, 255, 0.04) !important;
        color: #F4F4F5 !important;
    }}
    
    /* Active Link in Sidebar Navigation */
    div[data-testid="stSidebarNavItems"] ul li a[aria-current="page"] {{
        background-color: {colors['bg_trans']} !important;
        border: 1px solid {colors['border']} !important;
        color: {colors['primary']} !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stSidebarNavItems"] ul li a[aria-current="page"] span {{
        color: {colors['primary']} !important;
    }}
    
    /* Navigation Group Categories */
    div[data-testid="stSidebarNavSeparator"] {{
        border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
        margin: 16px 0 !important;
    }}
    span[data-testid="stSidebarNavLink-categoryName"] {{
        text-transform: uppercase !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        color: #52525B !important;
        letter-spacing: 0.08em !important;
        padding-left: 20px !important;
        margin-top: 16px !important;
        margin-bottom: 4px !important;
        display: block !important;
    }}

    /* Card components - Solid #18181B panels with crisp, thin borders */
    .info-card {{
        background-color: #18181B !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 18px !important;
        padding: 24px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
        transition: border-color 0.2s ease, transform 0.2s ease !important;
    }}
    .info-card:hover {{
        border-color: {colors['border']} !important;
        transform: translateY(-2px) !important;
    }}
    .header-card {{
        background-color: #18181B !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 18px !important;
        padding: 24px !important;
        margin-bottom: 24px !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    }}
    .header-card h3 {{
        color: #F4F4F5 !important;
        margin: 0 !important;
    }}
    .header-card p {{
        color: #A1A1AA !important;
        margin: 4px 0 0 0 !important;
    }}
    .info-card h1, .info-card h2, .info-card h3 {{
        color: #F4F4F5 !important;
        margin: 0 0 8px 0 !important;
    }}
    .info-card p {{
        color: #A1A1AA !important;
        margin: 0 !important;
        font-size: 15px !important;
    }}

    .bullet-card {{
        padding: 16px !important;
        margin-bottom: 12px !important;
        background-color: #18181B !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        transition: all 0.2s ease !important;
    }}
    .bullet-card:hover {{
        transform: translateX(4px) !important;
        border-color: {colors['border']} !important;
    }}
    .bullet-card strong {{
        color: #F4F4F5 !important;
    }}
    .bullet-card i {{
        color: #E4E4E7 !important;
        font-size: 13.5px !important;
    }}

    .b-strong {{
        border-left: 6px solid #10B981 !important;
    }}
    .b-weak {{
        border-left: 6px solid #EF4444 !important;
    }}
    .b-sugg {{
        color: {colors['primary']} !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        background: {colors['bg_trans']} !important;
        padding: 4px 8px !important;
        border-radius: 6px !important;
        display: inline-block !important;
        margin-top: 8px !important;
    }}

    /* Custom skill tags - clean flat design */
    .skill-tag {{
        display: inline-block !important;
        padding: 6px 14px !important;
        border-radius: 16px !important;
        margin: 4px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }}
    .skill-match {{
        background-color: rgba(16, 185, 129, 0.12) !important;
        color: #34D399 !important;
        border: 1px solid rgba(16, 185, 129, 0.25) !important;
    }}
    .skill-miss {{
        background-color: rgba(239, 68, 68, 0.12) !important;
        color: #F87171 !important;
        border: 1px solid rgba(239, 68, 68, 0.25) !important;
    }}

    /* Main Heading Title styling */
    .gradient-title {{
        color: #FAFAFA !important;
        font-weight: 800 !important;
        font-size: 3.25rem !important; /* Large, bold matching screenshot scale */
        letter-spacing: -0.03em !important;
        margin-bottom: 8px !important;
    }}

    /* Subheading text styling */
    .sub-heading {{
        color: #A1A1AA !important;
        font-size: 1.1rem !important;
        font-weight: 400 !important;
        margin-top: 0px !important;
        margin-bottom: 28px !important;
        max-width: 750px !important;
        line-height: 1.6 !important;
    }}

    /* Pill buttons matching the screenshot exactly */
    div.stButton > button {{
        background-color: {colors['primary']} !important;
        color: #000000 !important; /* Pure black text on bright colors for maximum clarity */
        border: none !important;
        border-radius: 9999px !important; /* Full pill shape */
        font-weight: 700 !important;
        padding: 12px 28px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px {colors['bg_trans']} !important;
        font-size: 15px !important;
    }}
    div.stButton > button:hover {{
        transform: translateY(-1px) !important;
        filter: brightness(1.1) !important;
        box-shadow: 0 6px 20px {colors['bg_trans']} !important;
    }}

    /* Secondary outline buttons (e.g., clear, file remove) */
    div.stButton > button[data-testid="stBaseButton-secondary"] {{
        background-color: transparent !important;
        color: #F4F4F5 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        box-shadow: none !important;
    }}
    div.stButton > button[data-testid="stBaseButton-secondary"]:hover {{
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }}

    /* Safe text styling for standard inputs (prevent conflicts) */
    .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {{
        background-color: #18181B !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        color: #F4F4F5 !important;
    }}
    
    /* Dropdown select popups */
    ul[role="listbox"] {{
        background-color: #18181B !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }}
    ul[role="listbox"] li {{
        background-color: transparent !important;
        color: #F4F4F5 !important;
    }}
    ul[role="listbox"] li:hover {{
        background-color: {colors['bg_trans']} !important;
        color: {colors['primary']} !important;
    }}

    /* Streamlit tabs */
    .stTabs [data-baseweb="tab"] {{
        color: #A1A1AA !important;
        font-weight: 600 !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {colors['primary']} !important;
        border-bottom-color: {colors['primary']} !important;
    }}
    
    /* File uploader wrapper and dropzones */
    [data-testid="stFileUploader"] {{
        background-color: rgba(24, 24, 27, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 16px !important;
        min-height: 10px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: rgba(34, 197, 94, 0.4) !important;
        box-shadow: 0 8px 32px rgba(34, 197, 94, 0.1) !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background-color: rgba(0, 0, 0, 0.2) !important;
        border: 2px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        padding: 20px 20px !important; /* Shorter vertically */
        min-height: 10px !important; /* Destroy native height */
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {colors['primary']} !important;
        background-color: {colors['bg_trans']} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        background-color: {colors['primary']} !important;
        color: #000000 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 32px !important;
        margin: 16px auto !important; /* Auto margin for perfect centering */
        display: block !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stFileUploaderDropzone"] button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px {colors['bg_trans']} !important;
    }}
    [data-testid="stFileUploaderDropzone"] small {{
        font-size: 14px !important;
        text-align: center !important;
        display: block !important;
        width: 100% !important;
    }}
    
    /* Kanban column layout customization */
    .kanban-col {{
        background-color: #121214 !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
    }}
    
    /* Warnings / Alerts */
    .stWarning {{
        background-color: rgba(245, 158, 11, 0.08) !important;
        border: 1px solid rgba(245, 158, 11, 0.15) !important;
        color: #FBBF24 !important;
        border-radius: 8px !important;
    }}
    
    .stSuccess {{
        background-color: rgba(16, 185, 129, 0.08) !important;
        border: 1px solid rgba(16, 185, 129, 0.15) !important;
        color: #34D399 !important;
        border-radius: 8px !important;
    }}
    
    /* Job Cards styling matching the clean dark layout */
    .job-card {{
        background-color: #18181B !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-left: 6px solid {colors['primary']} !important;
        border-radius: 18px !important;
        padding: 24px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }}
    .job-card:hover {{
        transform: translateY(-2px) !important;
        border-color: {colors['border']} !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3) !important;
    }}
    .job-title {{
        margin: 0 0 6px 0 !important; 
        color: #F4F4F5 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
    }}
    .job-meta {{
        margin: 5px 0 12px 0 !important; 
        color: #A1A1AA !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }}
    .job-desc {{
        color: #D4D4D8 !important; 
        font-size: 15.5px !important;
        line-height: 1.7 !important;
        margin: 0 !important;
    }}

    /* Pure CSS Expandable text box for clean descriptions */
    .desc-trigger {{
        display: none !important;
    }}
    .desc-wrapper {{
        max-height: 72px !important;
        overflow: hidden !important;
        transition: max-height 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
    }}
    /* Gradient fade-out on collapsed state */
    .desc-wrapper::after {{
        content: "" !important;
        position: absolute !important;
        bottom: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 32px !important;
        background: linear-gradient(to top, #18181B 0%, rgba(24, 24, 27, 0) 100%) !important;
        pointer-events: none !important;
        transition: opacity 0.2s ease !important;
    }}
    .desc-trigger:checked ~ .desc-wrapper {{
        max-height: 1000px !important;
    }}
    .desc-trigger:checked ~ .desc-wrapper::after {{
        opacity: 0 !important;
        pointer-events: none !important;
    }}
    .desc-btn {{
        display: inline-block !important;
        color: {colors['primary']} !important;
        cursor: pointer !important;
        font-size: 13.5px !important;
        font-weight: 700 !important;
        margin-top: 8px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        user-select: none !important;
    }}
    .desc-btn::before {{
        content: "Show More ▾" !important;
    }}
    .desc-trigger:checked ~ .desc-btn::before {{
        content: "Show Less ▴" !important;
    }}

    hr {{
        border: 0;
        height: 1px;
        background-color: rgba(255, 255, 255, 0.06);
        margin: 24px 0;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

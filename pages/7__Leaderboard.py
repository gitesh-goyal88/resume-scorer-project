import streamlit as st
import pandas as pd
from database import get_leaderboard_data
from ui_utils import inject_custom_css
inject_custom_css()

st.markdown("<h1 class='gradient-title' style='font-size: 2.8rem; margin-bottom: 5px; padding-bottom: 10px;'> Global Candidate Leaderboard</h1>", unsafe_allow_html=True)
st.markdown("Rankings are computed using **Industry Centroid Cosine Similarity** combined with **NLP Heuristic Health Scores**.")

if 'user_id' not in st.session_state or st.session_state.user_id is None:
    st.warning("Please login from the home page to view the leaderboard.")
    st.stop()

# Get all unique domains to populate the filter
# We fetch all data first to find domains
all_data = get_leaderboard_data("All")
domains = ["All Fields"] + sorted(list(set([row["Domain"] for row in all_data])))

col1, col2 = st.columns([1, 2])
with col1:
    selected_domain = st.selectbox("Filter by ML Predicted Domain:", domains)

# Fetch filtered data
db_filter = "All" if selected_domain == "All Fields" else selected_domain
leaderboard_data = get_leaderboard_data(db_filter)

if not leaderboard_data:
    st.info(f"No candidates found for domain: {selected_domain}")
    st.stop()

df = pd.DataFrame(leaderboard_data)
# Add rank column
df.index = df.index + 1
df.reset_index(inplace=True)
df.rename(columns={"index": "Rank"}, inplace=True)

# Highlight current user
def highlight_current_user(row):
    if row['Name'] == st.session_state.get('user_name', ''):
        return ['background-color: #DBEAFE; color: #1E3A8A; font-weight: bold'] * len(row)
    return [''] * len(row)

st.dataframe(
    df.style.apply(highlight_current_user, axis=1).format({
        "Health Score": "{:.0f}",
        "Centroid Score": "{:.2f}",
        "Total Score": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")
st.markdown("""
###  How is the Score Calculated?
1. **Centroid Score (60%):** We take your resume and calculate its TF-IDF Vector. We then compute the Cosine Similarity against the *Industry Centroid* (the mathematical average of all job descriptions in your predicted domain).
2. **Health Score (40%):** A rule-based NLP engine scores your formatting, bullet point length, and action verbs.
""")

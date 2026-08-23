import streamlit as st
import tempfile
import os
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import re
import io

from streamlit_lottie import st_lottie
from analyzer import (
    extract_text_from_pdf, extract_skills, check_formatting,
    compute_general_score, get_market_skill_gaps,
    extract_bullet_points, compute_section_scores, categorize_skills,
    extract_resume_features, calculate_yoe
)
from ml_model import (
    train_all_models,
    predict_job_category,
    classify_bullets,
    compute_health_score
)
from resume_builder import generate_interview_questions, generate_enhanced_pdf
from database import insert_candidate, save_user_resume, get_user_resumes
from job_matcher import get_domain_centroid_score
import shutil
from ui_utils import inject_custom_css
inject_custom_css()

# ── Session state init ─────────────────────────────────────────────────────────
for key in ["resume_text", "resume_path", "skills", "predicted_role", "issues",
            "bullet_results", "market_gaps", "section_scores", "ats_ml_score",
            "interview_qs", "answers", "db_saved", "health_data",]:
    if key not in st.session_state:
        st.session_state[key] = None

def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except:
        return None

lottie_ai = load_lottieurl("https://lottie.host/804d9c73-ec14-41e9-911b-c662a5bafbe5/2iPZJ29Npe.json")

# ── Chart Functions (LIGHT MODE) ──────────────────────────────────────────────
def create_gauge_chart(score, title="Score"):
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('white')
    angle = np.pi * (1 - score / 100)
    theta_bg = np.linspace(0, np.pi, 100)
    ax.fill_between(theta_bg, 0.6, 1.0, color='#E2E8F0', alpha=1.0)
    theta_score = np.linspace(np.pi, angle, 100)
    color = '#10B981' if score >= 70 else '#F59E0B' if score >= 40 else '#EF4444'
    ax.fill_between(theta_score, 0.6, 1.0, color=color, alpha=1.0)
    ax.set_ylim(0, 1.2)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.axis('off')
    ax.text(np.pi/2, 0.2, f"{score}", ha='center', va='center', fontsize=28, fontweight='bold', color=color)
    ax.text(np.pi/2, -0.15, title, ha='center', va='center', fontsize=12, color='#1E293B', fontweight='bold')
    plt.tight_layout()
    return fig

def create_radar_chart(skill_categories):
    categories = list(skill_categories.keys())
    values = [skill_categories[c]["score"] for c in categories]
    N = len(categories)
    if N == 0: return plt.subplots(figsize=(4,4))[0]
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.plot(angles, values, 'o-', linewidth=2, color='#2563EB')
    ax.fill(angles, values, alpha=0.2, color='#2563EB')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10, color='#1E293B')
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], size=7, color='#64748B')
    ax.spines['polar'].set_color('#E2E8F0')
    ax.grid(color='#E2E8F0', linewidth=1)
    plt.tight_layout()
    return fig

def create_section_bar_chart(section_scores):
    sections = list(section_scores.keys())
    scores = list(section_scores.values())
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    colors = ['#10B981' if s >= 70 else '#F59E0B' if s >= 40 else '#EF4444' for s in scores]
    bars = ax.barh(sections, scores, color=colors, height=0.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, f'{score}%', va='center', ha='left', fontsize=10, color='#1E293B', fontweight='bold')
    ax.set_xlim(0, 110)
    ax.tick_params(colors='#1E293B', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#E2E8F0')
    ax.spines['left'].set_color('#E2E8F0')
    plt.tight_layout()
    return fig

def create_word_cloud(text):
    stopwords = set(["and", "the", "to", "of", "in", "for", "with", "a", "on", "by", "an", "as", "at", "from"])
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues', stopwords=stopwords).generate(text)
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('white')
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    return fig

# ── Helper: Action Verbs & Salary ──────────────────────────────────────────────
def suggest_action_verbs(text):
    passive = ["worked on", "helped", "assisted", "was responsible for", "did", "made"]
    suggestions = []
    text_lower = text.lower()
    for p in passive:
        if p in text_lower:
            suggestions.append(p)
    return suggestions

def estimate_salary(role, ats_score):
    base = 60000
    if "Engineer" in role or "Developer" in role or "Data" in role:
        base = 80000
    elif "Manager" in role or "Lead" in role:
        base = 100000
    
    # Scale based on ATS score (proxy for quality/experience)
    multiplier = (ats_score / 100) + 0.5 # 0.5 to 1.5 range
    low = int((base * multiplier) / 1000) * 1000
    high = int((base * multiplier * 1.3) / 1000) * 1000
    return f"₹{low:,} - ₹{high:,}"


# ── Main UI: Hero Landing Page ──────────────────────────────────────────────────
st.markdown("""
<style>
/* Custom Hero Animations & Styling */
@keyframes glow {
    0% { box-shadow: 0 0 15px rgba(34, 197, 94, 0.2); }
    50% { box-shadow: 0 0 25px rgba(34, 197, 94, 0.6); }
    100% { box-shadow: 0 0 15px rgba(34, 197, 94, 0.2); }
}
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: 99px;
    color: #22C55E;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 24px;
    animation: glow 3s infinite;
}
div.hero-title, .stMarkdown div.hero-title {
    font-size: clamp(2.5rem, 6vw, 4.5rem) !important;
    font-weight: 800 !important;
    color: #FAFAFA !important;
    line-height: 1.1 !important;
    margin-bottom: 24px !important;
    text-align: center;
}
div.hero-title span, .stMarkdown div.hero-title span {
    font-size: inherit !important;
    background: linear-gradient(90deg, #10B981, #3B82F6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
p.hero-subtitle, .stMarkdown p.hero-subtitle {
    font-size: 1.15rem !important;
    color: #94A3B8 !important;
    text-align: center !important;
    max-width: 750px !important;
    margin: 0 auto 48px auto !important;
    line-height: 1.6 !important;
}
.hero-container {
    text-align: center;
    margin-top: -30px;
    position: relative;
    z-index: 1;
}
.feature-card {
    background: #18181B;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    height: 300px; /* Force strict uniform sizing */
    display: flex;
    flex-direction: column;
    transition: transform 0.2s, border-color 0.2s;
}
@media (max-width: 768px) {
    .feature-card {
        height: auto;
        min-height: 240px;
    }
}
.feature-card:hover {
    transform: translateY(-4px);
    border-color: rgba(34, 197, 94, 0.4);
}
.step-number {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #22C55E;
    color: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 14px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# 1. Hero Section
c1, c2, c3 = st.columns([1, 6, 1])
with c2:
    st.markdown("""
    <div class='hero-container'>
        <div class='hero-pill'> AI-powered candidate optimization</div>
        <div class='hero-title'>Build a resume that<br><span>outsmarts the ATS</span></div>
        <p class='hero-subtitle'>ResumeIQ translates your experience into structured data and recommends the highest-matching roles using cosine similarity across a dataset of 2,800+ real job postings.</p>
    </div>
    """, unsafe_allow_html=True)

    # Uploader natively centered & wider horizontally
    u1, u2, u3 = st.columns([1, 6, 1])
    with u2:
        st.markdown("<p style='color: #FAFAFA; font-weight: 600; font-size: 1.1rem; margin-bottom: 12px; text-align: center;'>Upload your resume to start</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
        status_placeholder = st.empty()
    # 3-Column Glassmorphism Stats Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display: flex; justify-content: center; gap: 24px; margin-top: 24px; flex-wrap: wrap;'>
        <div style='background: rgba(24, 24, 27, 0.4); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.05); border-radius: 16px; padding: 24px 32px; text-align: center; width: 180px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2); transition: all 0.3s ease;' onmouseover="this.style.borderColor='rgba(255,255,255,0.1)'" onmouseout="this.style.borderColor='rgba(255,255,255,0.05)'">
            <h3 style='margin: 0; font-size: 2.2rem; color: #FAFAFA; font-weight: 800;'>4K+</h3>
            <p style='margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem; font-weight: 500;'>Resumes Trained</p>
        </div>
        <div style='background: rgba(24, 24, 27, 0.4); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.05); border-radius: 16px; padding: 24px 32px; text-align: center; width: 180px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2); transition: all 0.3s ease;' onmouseover="this.style.borderColor='rgba(255,255,255,0.1)'" onmouseout="this.style.borderColor='rgba(255,255,255,0.05)'">
            <h3 style='margin: 0; font-size: 2.2rem; color: #FAFAFA; font-weight: 800;'>2.8K+</h3>
            <p style='margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem; font-weight: 500;'>Real Jobs</p>
        </div>
        <div style='background: rgba(24, 24, 27, 0.4); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.05); border-radius: 16px; padding: 24px 32px; text-align: center; width: 180px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2); transition: all 0.3s ease;' onmouseover="this.style.borderColor='rgba(255,255,255,0.1)'" onmouseout="this.style.borderColor='rgba(255,255,255,0.05)'">
            <h3 style='margin: 0; font-size: 2.2rem; color: #FAFAFA; font-weight: 800;'>92%</h3>
            <p style='margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem; font-weight: 500;'>Model Accuracy</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced ML Metrics Popover
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns([1, 2, 1])
    with m2:
        with st.popover(" View Advanced ML Metrics", use_container_width=True):
            st.markdown("""
            **Job Role Classifier Engine (60-40 Split)**
            - **Train Accuracy:** `95.32%`
            - **Test Accuracy:** `91.69%`
            - **Precision:** `96.15%`
            - **Recall:** `89.78%`
            - **F1-score:** `90.74%`
            """)

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

st.markdown("<br><br><br>", unsafe_allow_html=True)

# 2. How it works pipeline
st.markdown("<h2 style='text-align: center; font-size: 2rem; color: #FAFAFA; margin-bottom: 8px;'>How it works</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8; margin-bottom: 40px;'>Every resume flows through our intelligent pipeline</p>", unsafe_allow_html=True)

p1, p2, p3, p4 = st.columns(4)
with p1:
    st.markdown("""
    <div class='feature-card'>
        <div class='step-number'>1</div>
        <h4 style='color: #FAFAFA; margin-bottom: 8px;'>You upload</h4>
        <p style='color: #94A3B8; font-size: 14px;'>Upload your PDF resume. We extract the raw text and analyze formatting.</p>
    </div>
    """, unsafe_allow_html=True)
with p2:
    st.markdown("""
    <div class='feature-card'>
        <div class='step-number'>2</div>
        <h4 style='color: #FAFAFA; margin-bottom: 8px;'>AI interprets</h4>
        <p style='color: #94A3B8; font-size: 14px;'>Advanced TF-IDF pipelines extract deep technical skills, metrics, and exact experience depth.</p>
    </div>
    """, unsafe_allow_html=True)
with p3:
    st.markdown("""
    <div class='feature-card'>
        <div class='step-number'>3</div>
        <h4 style='color: #FAFAFA; margin-bottom: 8px;'>Engine matches</h4>
        <p style='color: #94A3B8; font-size: 14px;'>BM25 ranking and KNN algorithms instantly map your profile against 10,000+ real-world tech jobs.</p>
    </div>
    """, unsafe_allow_html=True)
with p4:
    st.markdown("""
    <div class='feature-card'>
        <div class='step-number'>4</div>
        <h4 style='color: #FAFAFA; margin-bottom: 8px;'>You explore</h4>
        <p style='color: #94A3B8; font-size: 14px;'>Get highly explainable insights with heatmaps, skill gaps, and interview prep.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br><br>", unsafe_allow_html=True)

# 3. Built for Career Growth Grid
st.markdown("<h2 style='text-align: center; font-size: 2rem; color: #FAFAFA; margin-bottom: 40px;'>Built for career growth</h2>", unsafe_allow_html=True)
g1, g2, g3 = st.columns(3)
with g1:
    st.markdown("""
    <div class='feature-card'>
        <h3 style='margin:0; margin-bottom:12px;'> Explainable AI</h3>
        <p style='color: #94A3B8; font-size: 14px;'>Every recommendation comes with a detailed analysis panel explaining exactly why you matched.</p>
    </div>
    """, unsafe_allow_html=True)
with g2:
    st.markdown("""
    <div class='feature-card'>
        <h3 style='margin:0; margin-bottom:12px;'> Interactive Prep</h3>
        <p style='color: #94A3B8; font-size: 14px;'>Simulate real-world technical and behavioral interviews using specialized AI personas.</p>
    </div>
    """, unsafe_allow_html=True)
with g3:
    st.markdown("""
    <div class='feature-card'>
        <h3 style='margin:0; margin-bottom:12px;'> Heatmap Analytics</h3>
        <p style='color: #94A3B8; font-size: 14px;'>Visualize your exact keyword strengths and weaknesses across 6 major tech domains.</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
g4, g5, g6 = st.columns(3)
with g4:
    st.markdown("""
    <div class='feature-card'>
        <h3 style='margin:0; margin-bottom:12px;'> LaTeX Templating</h3>
        <p style='color: #94A3B8; font-size: 14px;'>Instantly convert your plain text into beautiful, ATS-compliant PDF templates.</p>
    </div>
    """, unsafe_allow_html=True)
with g5:
    st.markdown("""
    <div class='feature-card'>
        <h3 style='margin:0; margin-bottom:12px;'> BM25 & KNN Engine</h3>
        <p style='color: #94A3B8; font-size: 14px;'>Lightning-fast candidate matching powered by advanced BM25 ranking and K-Nearest Neighbors.</p>
    </div>
    """, unsafe_allow_html=True)
with g6:
    st.markdown("""
    <div class='feature-card'>
        <h3 style='margin:0; margin-bottom:12px;'> Instant Feedback</h3>
        <p style='color: #94A3B8; font-size: 14px;'>Grammar checks, action verb counting, and formatting validation in milliseconds.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br><br>", unsafe_allow_html=True)

# 4. Bottom CTA
st.markdown("""
<style>
.glass-cta-wrapper {
    background: rgba(24, 24, 27, 0.4);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 60px 24px;
    text-align: center;
    margin: 40px auto;
    max-width: 800px;
    margin-bottom: -110px; /* Pull the next element up! */
    padding-bottom: 120px; /* Make room inside the box for the button */
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}
</style>
<div class='glass-cta-wrapper'>
    <h2 style='font-size: 2.5rem; color: #FAFAFA; margin-bottom: 16px; font-weight: 700;'>Ready to optimize your career?</h2>
    <p style='color: #94A3B8; font-size: 1.1rem; max-width: 500px; margin: 0 auto; line-height: 1.6;'>Upload your resume, unlock AI-driven insights, and start mapping your skills to the top tech roles.</p>
</div>
""", unsafe_allow_html=True)

cta1, cta2, cta3 = st.columns([1.5, 2, 1.5])
with cta2:
    if st.button(" Let's Try - Go to Resume Analysis", use_container_width=True, type="primary"):
        st.switch_page("pages/2__Resume_Analysis.py")

st.markdown("<br><br><br>", unsafe_allow_html=True)

# Execution of upload logic:
if uploaded_file and not st.session_state.get("resume_text"):
    with status_placeholder.status(" Analyzing your resume pipeline...", expanded=True) as status:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        st.write(" Parsing PDF structure...")
        text = extract_text_from_pdf(tmp_path)
        from resume_builder import clean_resume_text_bullets
        text = clean_resume_text_bullets(text)
        st.session_state.resume_text = text
        st.session_state.resume_path = tmp_path
        
        st.write(" Extracting Technical Skills...")
        skills = extract_skills(text)
        st.session_state.skills = skills
        
        st.write(" Checking Formatting & Grammar...")
        issues = check_formatting(text, tmp_path)
        st.session_state.issues = issues
        st.session_state.health_data = compute_general_score(text, issues, skills)
        st.session_state.section_scores = compute_section_scores(text)
        
        st.write(" Detecting Job Role...")
        prediction = predict_job_category(text)
        st.session_state.predicted_role = prediction["category"]
        
        st.write(" Llama-3 Bullet Inference...")
        st.session_state.bullet_results = classify_bullets(extract_bullet_points(text))
        st.session_state.yoe = calculate_yoe(text)
        
        st.write(" Computing TF-IDF Heatmaps...")
        gaps = get_market_skill_gaps(st.session_state.predicted_role, skills)
        st.session_state.market_gaps = gaps
        
        features = extract_resume_features(text, tmp_path)
        raw_features = {
            "skill_count":        len(st.session_state.skills or []),
            "tfidf_skill_score":  features.get("tfidf_skill_score", 0),
            "action_verb_count":  features.get("action_verb_count", 0),
            "metrics_count":      features.get("metrics_count", 0),
            "section_count":      features.get("section_completeness", 0) // 25,
            "formatting_penalty": len([i for i in (st.session_state.issues or []) if i.get("severity") == "high"]),
        }
        health_result = compute_health_score(raw_features)
        base_ats = health_result["total_score"]
        st.session_state.ats_ml_score = {"score": base_ats, "grade": health_result["grade"]}
            
        status.update(label=" Analysis Complete! Redirecting...", state="complete", expanded=False)
        
    st.switch_page("pages/2__Resume_Analysis.py")

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
            "interview_qs", "answers", "db_saved", "health_data",
            "edit_name", "edit_summary", "edit_skills_languages", "edit_skills_tools", "edit_skills_soft", "edit_experience", 
            "edit_education", "edit_achievements", "edit_certs_projects",
            "edit_state_initialized"]:
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

# ── Chart Functions (DARK MODE) ──────────────────────────────────────────────
def create_gauge_chart(score, title="Score"):
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#18181B')
    ax.set_facecolor('#18181B')
    angle = np.pi * (1 - score / 100)
    theta_bg = np.linspace(0, np.pi, 100)
    ax.fill_between(theta_bg, 0.6, 1.0, color='#2A2A2E', alpha=1.0)
    theta_score = np.linspace(np.pi, angle, 100)
    color = '#22C55E' if score >= 70 else '#F59E0B' if score >= 40 else '#EF4444'
    ax.fill_between(theta_score, 0.6, 1.0, color=color, alpha=1.0)
    ax.set_ylim(0, 1.2)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.axis('off')
    ax.text(np.pi/2, 0.2, f"{score}", ha='center', va='center', fontsize=28, fontweight='bold', color=color)
    ax.text(np.pi/2, -0.15, title, ha='center', va='center', fontsize=12, color='#FAFAFA', fontweight='bold')
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
    fig.patch.set_facecolor('#18181B')
    ax.set_facecolor('#18181B')
    ax.plot(angles, values, 'o-', linewidth=2, color='#22C55E')
    ax.fill(angles, values, alpha=0.2, color='#22C55E')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10, color='#FAFAFA')
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], size=7, color='#A1A1AA')
    ax.spines['polar'].set_color('#3B3B40')
    ax.grid(color='#2A2A2E', linewidth=1)
    plt.tight_layout()
    return fig

def create_section_bar_chart(section_scores):
    sections = list(section_scores.keys())
    scores = list(section_scores.values())
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor('#18181B')
    ax.set_facecolor('#18181B')
    colors = ['#22C55E' if s >= 70 else '#F59E0B' if s >= 40 else '#EF4444' for s in scores]
    bars = ax.barh(sections, scores, color=colors, height=0.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, f'{score}%', va='center', ha='left', fontsize=10, color='#FAFAFA', fontweight='bold')
    ax.set_xlim(0, 110)
    ax.tick_params(colors='#FAFAFA', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#3B3B40')
    ax.spines['left'].set_color('#3B3B40')
    plt.tight_layout()
    return fig

def create_word_cloud(text):
    stopwords = set(["and", "the", "to", "of", "in", "for", "with", "a", "on", "by", "an", "as", "at", "from"])
    wordcloud = WordCloud(width=800, height=400, background_color='#18181B', colormap='Greens', stopwords=stopwords).generate(text)
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#18181B')
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

# ── Main UI ────────────────────────────────────────────────────────────────────
# UI Customization styles are injected globally from ui_utils.py
st.markdown("<h1 class='gradient-title' style='font-size: 3rem; margin-bottom: 5px; padding-bottom: 5px;'> Resume Analysis</h1>", unsafe_allow_html=True)

col_title1, col_title2 = st.columns([0.7, 0.3])
with col_title1:
    st.markdown("<p class='sub-heading'>Your comprehensive AI analysis report, detailed metrics, and live editor.</p>", unsafe_allow_html=True)
with col_title2:
    if st.button(" Clear Cache & Restart", use_container_width=True, help="Click to wipe memory and force the new ML Engine to run"):
        keys_to_keep = ["user_id", "logged_in", "username"]
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.switch_page("pages/1__Dashboard.py")

if not st.session_state.resume_text or not st.session_state.ats_ml_score:
    st.markdown('''
    <div style='background: #18181B; border: 1px dashed rgba(255,255,255,0.15); border-radius: 18px; padding: 40px; text-align: center; margin-top: 20px;'>
        <h3 style='color: #FAFAFA; margin-bottom: 8px;'>No Analysis Available</h3>
        <p style='color: #A1A1AA; font-family: Inter; margin-bottom: 24px;'>Please upload a resume below to begin analysis.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file:
        import tempfile
        from analyzer import extract_text_from_pdf
        from ml_model import (
            extract_skills, predict_job_category, classify_bullets,
            extract_bullet_points, get_market_skill_gaps, calculate_yoe,
            extract_resume_features, compute_health_score
        )
        from resume_builder import clean_resume_text_bullets
        from formatting_checker import check_formatting, compute_general_score, compute_section_scores
        
        with st.status(" Analyzing your resume pipeline...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            st.write(" Parsing PDF structure...")
            text = extract_text_from_pdf(tmp_path)
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
                
            status.update(label=" Analysis Complete! Reloading...", state="complete", expanded=False)
            
        st.rerun()
    st.stop()

if st.session_state.resume_text:
        
    role = st.session_state.predicted_role or "Professional"
    ats = int((st.session_state.ats_ml_score or {"score": 0})["score"])
    
    # DB Save
    if not st.session_state.db_saved:
        try:
            email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", st.session_state.resume_text)
            email = email_match.group(0) if email_match else "unknown@email.com"
            name = st.session_state.resume_text.splitlines()[0][:30]
            health = st.session_state.health_data["score"]
            strong = sum(1 for b in st.session_state.bullet_results if b["label"] == "Strong")
            insert_candidate(name, email, "", role, ats, health, strong)
            
            # Save resume to user account if logged in
            if st.session_state.get('user_id'):
                os.makedirs("data/saved_resumes", exist_ok=True)
                saved_path = f"data/saved_resumes/{st.session_state.user_id}_{int(__import__('time').time())}.pdf"
                if st.session_state.resume_path and os.path.exists(st.session_state.resume_path):
                    shutil.copy2(st.session_state.resume_path, saved_path)
                
                # Compute ML Centroid Score
                centroid = get_domain_centroid_score(st.session_state.resume_text, role)
                save_user_resume(st.session_state.user_id, saved_path, False, health, role, centroid)
            
            st.session_state.db_saved = True
        except:
            pass
    
    # --- My Saved Resumes Gallery ---
    if st.session_state.get('user_id'):
        saved_resumes = get_user_resumes(st.session_state.user_id)
        if saved_resumes:
            with st.expander(f" My Saved Resumes ({len(saved_resumes)}/4)", expanded=False):
                for i, res in enumerate(saved_resumes):
                    badge = " Enhanced" if res['is_enhanced'] else " Original"
                    rcol1, rcol2, rcol3 = st.columns([3, 1, 1])
                    with rcol1:
                        st.markdown(f"**{badge}** — {res['domain']} — Uploaded: {res['upload_date'][:10]}")
                    with rcol2:
                        st.markdown(f"Health: **{res['health_score']}** | Centroid: **{res['centroid_score']}**")
                    with rcol3:
                        if os.path.exists(res['file_path']):
                            with open(res['file_path'], 'rb') as rf:
                                st.download_button("", rf.read(), file_name=f"resume_{i+1}.pdf", key=f"dl_saved_{i}")
                    st.markdown("---")

    st.markdown("---")
    
    # LinkedIn & Salary Card
    salary = estimate_salary(role, ats)
    st.markdown(f"""
    <div class='info-card'>
        <h1 style='color: #FAFAFA; font-size: 2.5rem; margin-bottom: 5px;'>Resume Health Score</h1>
        <p style='color: #A1A1AA; font-size: 1.1rem; margin-top: 0;'>A comprehensive analysis of your resume's impact and readability.</p>
        <h3 style='margin-top:0; color:#22C55E;'>Estimated Market Value: {salary}</h3>
    </div>
    """, unsafe_allow_html=True)

    # Compile current active skills to reflect manual edits in the editor
    active_skills = []
    if "edit_skills_languages_list" in st.session_state and st.session_state.edit_skills_languages_list is not None:
        active_skills.extend([s.strip() for s in st.session_state.edit_skills_languages_list if s.strip()])
    if "edit_skills_tools_list" in st.session_state and st.session_state.edit_skills_tools_list is not None:
        active_skills.extend([s.strip() for s in st.session_state.edit_skills_tools_list if s.strip()])
    if "edit_skills_soft_list" in st.session_state and st.session_state.edit_skills_soft_list is not None:
        active_skills.extend([s.strip() for s in st.session_state.edit_skills_soft_list if s.strip()])
    if not active_skills and st.session_state.get("skills"):
        active_skills = list(st.session_state.skills)
    active_skills = list(set(active_skills))

    @st.dialog("Your Auto-Generated LinkedIn Bio")
    def show_linkedin_bio():
        skills_str = ", ".join(active_skills[:5]) if active_skills else "problem-solving"
        bio = f"Driven and detail-oriented {role} with a proven track record of delivering high-quality results. Skilled in {skills_str}, I thrive in collaborative environments where I can leverage technology to solve complex problems.\n\nAlways eager to learn and adapt to new challenges, I am currently looking for opportunities to bring my expertise to an innovative team."
        st.write("Copy and paste this into your LinkedIn 'About' section:")
        st.code(bio, language="markdown")

    if st.button(" Generate LinkedIn Bio"):
        show_linkedin_bio()

    st.markdown("---")
    yoe = st.session_state.get("yoe", 0.0)
    st.markdown(f"<h2 style='color: #FAFAFA;'> AI Benchmark Report: {role} ({yoe} YoE)</h2>", unsafe_allow_html=True)
    
    #  Role override selectbox directly in the report view
    from analyzer import MARKET_SKILLS
    role_options = sorted(list(MARKET_SKILLS.keys()))
    if role not in role_options:
        role_options.append(role)
        role_options = sorted(role_options)
        
    selected_role = st.selectbox(
        " Target Job Role for Skills Benchmarking (change this if the ML classification is incorrect):",
        options=role_options,
        index=role_options.index(role)
    )
    if selected_role != role:
        st.session_state.predicted_role = selected_role
        st.rerun()
        
    st.markdown("Your resume is scored on three academic pillars: Impact, Presentation, and Competencies.")
    
    # Calculate VMock Pillars
    # 1. Presentation = Formatting Health
    presentation_score = st.session_state.health_data["score"]
    
    # 2. Competencies = ATS Market Skill Alignment (from gaps)
    # Dynamically compute gaps based on active skills and target role
    gaps = get_market_skill_gaps(st.session_state.predicted_role, active_skills)
    st.session_state.market_gaps = gaps
    
    competencies_score = 100
    if len(gaps["matched"]) + len(gaps["missing"]) > 0:
        competencies_score = int((len(gaps["matched"]) / (len(gaps["matched"]) + len(gaps["missing"]))) * 100)
        
    # 3. Impact = Ratio of Strong Bullets
    bullets = st.session_state.bullet_results
    strong_count = sum(1 for b in bullets if b["label"] == "Strong")
    impact_score = int((strong_count / max(1, len(bullets))) * 100)
    
    # 4. Use the actual ML-driven ATS Score from the backend!
    # Earlier in this file (or in Dashboard), we calculated health_result
    # and saved the final ML score to st.session_state.ats_ml_score.
    if "ats_ml_score" in st.session_state and st.session_state.ats_ml_score:
        combined_score = st.session_state.ats_ml_score.get("score", 0)
    else:
        combined_score = 0
        
    # We also need the raw tfidf_skill_score to show in the progress bar below.
    # Since we extract features upon upload, we'll re-extract here if needed,
    # or just fetch it from the UI's local recalculation.
    try:
        from analyzer import extract_resume_features
        _feats = extract_resume_features(st.session_state.resume_text, None)
        competencies_score = int(_feats.get("tfidf_skill_score", 0))
    except:
        competencies_score = 0
    
    st.markdown("###  ATS Readiness & ML Breakdown")
    col_main, col_sub = st.columns([0.4, 0.6], gap="large")
    
    with col_main:
        fig_combined = create_gauge_chart(combined_score, "Overall ATS Score")
        st.pyplot(fig_combined)
        plt.close(fig_combined)
        
    with col_sub:
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border-radius: 16px; padding: 30px; height: 100%; display: flex; flex-direction: column; justify-content: center; gap: 24px; margin-top: 15px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);">
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="color: #D4D4D8; font-weight: 600; font-size: 15px;"> Bullet Impact Score</span>
                    <span style="color: #FAFAFA; font-weight: 700;">{impact_score}/100</span>
                </div>
                <div style="background: rgba(255,255,255,0.06); border-radius: 99px; width: 100%; height: 10px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #3B82F6, #8B5CF6); width: {impact_score}%; height: 100%; border-radius: 99px; transition: width 1s ease-in-out;"></div>
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="color: #D4D4D8; font-weight: 600; font-size: 15px;"> Formatting Health</span>
                    <span style="color: #FAFAFA; font-weight: 700;">{presentation_score}/100</span>
                </div>
                <div style="background: rgba(255,255,255,0.06); border-radius: 99px; width: 100%; height: 10px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #10B981, #34D399); width: {presentation_score}%; height: 100%; border-radius: 99px; transition: width 1s ease-in-out;"></div>
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="color: #D4D4D8; font-weight: 600; font-size: 15px;"> Hard Skill Alignment</span>
                    <span style="color: #FAFAFA; font-weight: 700;">{competencies_score}/100</span>
                </div>
                <div style="background: rgba(255,255,255,0.06); border-radius: 99px; width: 100%; height: 10px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #F59E0B, #FBBF24); width: {competencies_score}%; height: 100%; border-radius: 99px; transition: width 1s ease-in-out;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("###  Explainable AI (XAI) Insights")
    xai_bullets = []
    if impact_score > 80:
        xai_bullets.append(" **High Impact:** Your Random Forest score was significantly boosted by a strong presence of Action Verbs and Metrics in your bullet points.")
    else:
        xai_bullets.append(" **Low Impact Penalty:** Your score was penalized due to a lack of quantifiable metrics. Adding numbers to your achievements will increase your score.")
        
    if presentation_score > 85:
        xai_bullets.append(" **Clean Presentation:** The parser easily extracted your data due to excellent formatting health.")
    else:
        xai_bullets.append(" **Formatting Penalty:** The ML engine struggled to parse some sections. Fix your margins or font consistency to prevent ATS rejection.")
        
    if competencies_score > 75:
        xai_bullets.append(" **Market Aligned:** You possess a high density of the hard skills expected for this specific role, boosting your Competencies score.")
    else:
        xai_bullets.append(" **Skill Gap Penalty:** Your resume lacks critical skills required for the current market, heavily weighing down your Resume Health Score.")

    for insight in xai_bullets:
        st.markdown(insight)

    st.markdown("---")
    
    # Row 2: Skills & Gaps
    col_gaps, col_radar = st.columns([1.5, 1])
    
    with col_radar:
        st.markdown("###  Skill Distribution")
        skill_cats = categorize_skills(active_skills)
        fig_radar = create_radar_chart(skill_cats)
        st.pyplot(fig_radar)
        plt.close(fig_radar)
        
        st.markdown("###  Keyword Cloud")
        fig_cloud = create_word_cloud(st.session_state.resume_text)
        st.pyplot(fig_cloud)
        plt.close(fig_cloud)

    with col_gaps:
        st.markdown("###  Market Gap & Skills Improvement")
        gaps = st.session_state.market_gaps
        if gaps["missing"]:
            st.error(" **Missing In-Demand Skills:** Add these to boost your Resume Health Score for this role.")
            missing_html = "".join([f"<span class='skill-tag skill-miss'>{s}</span>" for s in gaps["missing"]])
            st.markdown(missing_html, unsafe_allow_html=True)
        else:
            st.success(" No major market gaps detected for this role!")
            
        st.markdown(" **Your Top Skills:**")
        matched_html = "".join([f"<span class='skill-tag skill-match'>{s}</span>" for s in gaps["matched"][:10]])
        st.markdown(matched_html, unsafe_allow_html=True)
        
        st.markdown("###  Formatting Issues to Fix")
        if not st.session_state.issues:
            st.success("No formatting issues found!")
        else:
            for issue in st.session_state.issues:
                st.warning(f"**{issue['severity'].upper()}**: {issue['issue']}")

    st.markdown("---")
    
    # Live Upskilling Recommender
    st.header(" Automated Upskilling Recommender")
    st.markdown("Bridge your market gap. We've scraped the web for the top real-life courses for your missing skills.")
    
    if gaps["missing"]:
        from course_scraper import fetch_courses
        
        top_3_missing = gaps["missing"][:3]
        
        # Create 'Slide View' using tabs
        tabs = st.tabs(top_3_missing)
        
        for i, skill in enumerate(top_3_missing):
            with tabs[i]:
                st.markdown(f"**Top recommendations for:** `{skill}`")
                
                with st.spinner(f" Live scraping YouTube & Aggregators for '{skill}' courses..."):
                    courses = fetch_courses(skill, limit=3)
                    
                if not courses:
                    st.info(f"Could not fetch live courses for {skill}. Try checking Udemy manually.")
                else:
                    accent = st.session_state.get("theme_accent", "Green")
                    accent_colors = {
                        "Blue": "#60A5FA", "Green": "#1ed760", "Red": "#fb7185", "Purple": "#a78bfa", "Amber": "#facc15"
                    }
                    bg_trans = {
                        "Blue": "rgba(96, 165, 250, 0.15)", "Green": "rgba(30, 215, 96, 0.15)",
                        "Red": "rgba(251, 113, 133, 0.15)", "Purple": "rgba(167, 139, 250, 0.15)", "Amber": "rgba(250, 204, 21, 0.15)"
                    }
                    active_color = accent_colors.get(accent, "#1ed760")
                    active_bg = bg_trans.get(accent, "rgba(30, 215, 96, 0.15)")

                    cols = st.columns(3)
                    for j, course in enumerate(courses[:3]):
                        with cols[j]:
                            st.markdown(f"""
                            <div style='background-color: #18181B; border: 1px solid rgba(255, 255, 255, 0.08); padding: 20px; border-radius: 12px; height: 100%; box-shadow: 0 4px 15px rgba(0,0,0,0.25); display: flex; flex-direction: column; justify-content: space-between;'>
                                <div>
                                    <span style='font-size: 10px; color: {active_color}; font-weight: 700; background: {active_bg}; padding: 3px 8px; border-radius: 6px; text-transform: uppercase;'>{course['platform'].upper()}</span>
                                    <h5 style='color: #F4F4F5 !important; margin: 12px 0 16px 0; font-family: "Outfit", sans-serif; font-weight: 600; line-height: 1.4; font-size: 14px;'>{course['title'][:60]}{"..." if len(course['title'])>60 else ""}</h5>
                                </div>
                                <a href="{course['url']}" target="_blank" style='text-decoration: none; color: {active_color}; font-weight: 700; font-size: 13px; display: inline-flex; align-items: center; gap: 4px;'>Watch Course </a>
                            </div>
                            """, unsafe_allow_html=True)
    else:
        st.success("You are fully aligned with the required market skills! No urgent upskilling required.")

    st.markdown("---")
    
    # Row 3: ResumeWorded Line-by-Line Breakdown
    st.markdown("###  Line-by-Line Bullet Breakdown")
    st.markdown("We've extracted every bullet point on your resume. Here is granular, line-by-line feedback on your writing.")
    
    for b in bullets:
        text = b['text']
        ml_label = b.get("label", "Weak")
        
        feedback = []
        
        # 1. Prioritize Deep Learning Semantic Feedback
        if b.get("feedback"):
            feedback.append(b["feedback"])
            
        # 2. Fallback to Granular Regex Heuristics ONLY if ML gave no feedback
        elif ml_label == "Weak":
            # Check for action verb
            action_verbs = ["developed", "led", "managed", "created", "built", "improved", "designed", "optimized", "spearheaded", "implemented"]
            has_action = any(v in text.lower() for v in action_verbs)
            if not has_action:
                feedback.append("Missing strong action verb (e.g. 'led', 'developed').")
                
            # Only require a metric if the bullet describes an outcome/optimization/impact
            impact_words = ["increase", "decrease", "improve", "reduce", "grow", "save", "cut", "boost",
                            "revenue", "latency", "speed", "efficient", "optimize", "accelerate", "cost", "budget"]
            needs_metric = any(w in text.lower() for w in impact_words)
            has_metric = bool(re.search(r'\b\d+%\b|\$\d+|\b\d+\b', text))
            
            if needs_metric and not has_metric:
                feedback.append("Missing quantifiable metric (e.g. '20%', '$50k', '5+ developers').")
                
            if not feedback:
                feedback.append("Could be stronger. Detail the final outcome or scale of your contribution.")
                
        if ml_label == "Strong" and not feedback:
            st.markdown(f"<div class='bullet-card b-strong'> <strong>Perfect Impact</strong><br><i>\"{text}\"</i></div>", unsafe_allow_html=True)
        else:
            feedback_str = " | ".join(feedback)
            st.markdown(f"<div class='bullet-card b-weak'> <strong>Needs Work</strong><br><i>\"{text}\"</i><br><span class='b-sugg'> Feedback: {feedback_str}</span></div>", unsafe_allow_html=True)


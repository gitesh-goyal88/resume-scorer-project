import streamlit as st
import re
import os
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from analyzer import extract_skills
from resume_builder import generate_cover_letter_pdf
from llm_utils import call_groq_api




from ui_utils import inject_custom_css
inject_custom_css()

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    [data-testid="stAppDeployButton"] {display: none;}
    footer {visibility: hidden;}
    .metric-card { background: #18181B; border: 1px solid rgba(255, 255, 255, 0.08); padding: 24px; border-radius: 18px; text-align: center; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2); }
    .keyword-match { display: inline-block; padding: 4px 10px; border-radius: 12px; margin: 4px; font-size: 13px; font-weight: 500; background: rgba(16, 185, 129, 0.12); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.25); }
    .keyword-miss { display: inline-block; padding: 4px 10px; border-radius: 12px; margin: 4px; font-size: 13px; font-weight: 500; background: rgba(239, 68, 68, 0.12); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.25); }
</style>
""", unsafe_allow_html=True)

def highlight_keywords(text, matched_list):
    """Highlight matched keywords in the resume text using HTML mark tags."""
    highlighted = text
    # Clean matches (remove " (matched as ...)" descriptions)
    clean_matched = []
    for m in matched_list:
        if " (matched as " in m:
            clean_matched.append(m.split(" (matched as ")[0])
        else:
            clean_matched.append(m)
            
    # Sort matched words descending by length to avoid partial highlighting of substrings
    sorted_matched = sorted(list(set(clean_matched)), key=len, reverse=True)
    
    for m in sorted_matched:
        if not m.strip():
            continue
        try:
            pattern = re.compile(rf"\b({re.escape(m)})\b", re.IGNORECASE)
            highlighted = pattern.sub(r"<mark style='background-color: #FEF08A; color: black; padding: 2px 4px; border-radius: 4px; font-weight: bold;'>\1</mark>", highlighted)
        except Exception:
            # Fallback for complex strings
            highlighted = highlighted.replace(m, f"<mark style='background-color: #FEF08A; color: black; padding: 2px 4px; border-radius: 4px; font-weight: bold;'>{m}</mark>")
    return highlighted

if "resume_text" not in st.session_state or not st.session_state.resume_text:
    st.warning(" Please upload your resume in the Candidate Portal first.")
    st.stop()

st.title(" Custom Job Scanner")
st.markdown("Paste a Job Description (JD) below to see exactly how well your resume matches the required keywords.")

jd_text = st.text_area("Paste Job Description here:", height=250, placeholder="E.g. We are looking for a Software Engineer with experience in Python, AWS, and Docker...")

if st.button(" Scan Resume vs JD", type="primary"):
    if len(jd_text.split()) < 20:
        st.error("Please paste a longer job description for an accurate scan.")
    else:
        with st.spinner("Analyzing keyword overlap..."):
            # 1. Extract Skills from JD
            jd_skills = extract_skills(jd_text)
            resume_skills = st.session_state.get("skills", [])
            
            # 2. Find Missing vs Matched using Fuzzy Semantic Matching
            import difflib
            resume_skills_lower = [s.lower() for s in resume_skills]
            matched = []
            missing = []
            
            for skill in jd_skills:
                skill_lower = skill.lower()
                # Check for exact match first
                if skill_lower in resume_skills_lower:
                    matched.append(skill)
                else:
                    # Fuzzy match (threshold 0.75 matches "react" to "react.js")
                    close_matches = difflib.get_close_matches(skill_lower, resume_skills_lower, n=1, cutoff=0.75)
                    if close_matches:
                        matched.append(f"{skill} (matched as {close_matches[0]})")
                    else:
                        missing.append(skill)
                    
            # 3. Calculate Match Score based on Skill Overlap (Jobscan methodology)
            if len(jd_skills) > 0:
                skill_match_score = (len(matched) / len(jd_skills)) * 100
            else:
                skill_match_score = 0
                
            # Combine with baseline TF-IDF for general text similarity
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf = vectorizer.fit_transform([st.session_state.resume_text, jd_text])
            from sklearn.metrics.pairwise import cosine_similarity
            text_sim_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
            
            # 80% weight on exact skills, 20% on general text similarity
            if len(jd_skills) > 0:
                match_score = int((skill_match_score * 0.8) + (text_sim_score * 0.2))
            else:
                match_score = int(text_sim_score)
            
            match_score = min(100, match_score)
            
            # Display Results
            st.markdown("---")
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin:0; color:#F4F4F5;'>Resume Match Rate</h3>
                <h1 style='margin:10px 0; font-size:48px; color:{"#10B981" if match_score >= 70 else "#F59E0B" if match_score >= 40 else "#EF4444"};'>{match_score}%</h1>
                <p style='margin:0; color:#A1A1AA;'>Aim for 75%+ before submitting your application.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("###  Missing Keywords")
                st.markdown("Add these exact words to your resume to beat the ATS:")
                if not missing:
                    st.success("You have all the required skills!")
                else:
                    missing_html = "".join([f"<span class='keyword-miss'>{s}</span>" for s in missing])
                    st.markdown(missing_html, unsafe_allow_html=True)
            
            with col2:
                st.markdown("###  Matched Keywords")
                st.markdown("You already have these required skills:")
                if not matched:
                    st.error("No keywords matched.")
                else:
                    matched_html = "".join([f"<span class='keyword-match'>{s}</span>" for s in matched])
                    st.markdown(matched_html, unsafe_allow_html=True)

            # --- Visual Keyword Match Highlighter ---
            st.markdown("---")
            st.markdown("###  Highlighted Resume (Visual Keyword Matcher)")
            st.markdown("Below is your original resume text with matched keywords highlighted in yellow:")
            with st.expander("View Highlighted Resume Text", expanded=True):
                highlighted_resume = highlight_keywords(st.session_state.resume_text, matched)
                formatted_resume = highlighted_resume.replace("\n", "<br>")
                st.markdown(f"<div style='background-color: #F8FAFC; padding: 20px; border-radius: 8px; border: 1px solid #E2E8F0; font-family: monospace; font-size: 13px; color: black; line-height: 1.6;'>{formatted_resume}</div>", unsafe_allow_html=True)

            # --- Cold Outreach & Cover Letter generation ---
            st.markdown("---")
            st.markdown("###  Generate Outreach Assets & Cover Letter")
            st.markdown("Create cold outreach emails and custom cover letters tailored specifically to this job description.")
            
            c_out1, c_out2 = st.columns(2)
            with c_out1:
                job_title_input = st.text_input("Target Job Title:", value="Software Engineer")
            with c_out2:
                company_input = st.text_input("Target Company Name:", value="Google")
                
            tab_email, tab_cl = st.tabs([" Cold Outreach Email", " Cover Letter PDF"])
            
            with tab_email:
                st.write("Generate a short, high-impact cold message for LinkedIn or Email.")
                if st.button("Generate Cold Email", type="secondary"):
                    with st.spinner("Generating outreach email..."):
                        # Extract candidate name from first line of resume
                        candidate_name = st.session_state.resume_text.splitlines()[0][:30] if st.session_state.resume_text else "Candidate"
                        prompt = f"""Generate a cold outreach message.
Name: {candidate_name}
Job Title: {job_title_input}
Company: {company_input}
Job Description: {jd_text}
Candidate Resume: {st.session_state.resume_text[:2000]}"""
                        
                        system_prompt = "You are a professional recruiting assistant. Generate a highly polished, short cold email/message (70-100 words). Do not use placeholders."
                        outreach_content = call_groq_api(prompt, system_prompt)
                        st.markdown("**Your Custom Outreach Message:**")
                        st.code(outreach_content, language="markdown")
                        
            with tab_cl:
                st.write("Generate a formatted Cover Letter PDF using the matched keywords and your resume.")
                if st.button("Generate Cover Letter", type="secondary"):
                    with st.spinner("Compiling Cover Letter PDF..."):
                        cl_output_path = os.path.join(tempfile.gettempdir(), f"cover_letter_{company_input}.pdf")
                        generate_cover_letter_pdf(
                            st.session_state.resume_text,
                            [m.split(" (matched as ")[0] if " (matched as " in m else m for m in matched],
                            job_title_input,
                            company_input,
                            cl_output_path
                        )
                        st.success(f" Cover Letter generated for {company_input}!")
                        with open(cl_output_path, "rb") as f:
                            st.download_button(
                                label=" Download Cover Letter PDF",
                                data=f.read(),
                                file_name=f"cover_letter_{company_input}.pdf",
                                mime="application/pdf"
                            )


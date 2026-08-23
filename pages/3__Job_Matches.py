import streamlit as st
import os
import tempfile
from job_matcher import recommend_jobs
from resume_builder import generate_cover_letter_pdf
from ui_utils import inject_custom_css

st.session_state.jobs_viewed = True

inject_custom_css()




# UI Customization styles are injected globally from ui_utils.py

st.markdown("<h1 class='gradient-title' style='font-size: 3rem; margin-bottom: 5px; padding-bottom: 5px;'> Live Job Recommendations</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-heading'>Based on your uploaded resume, our TF-IDF matching engine has found the best active roles for you.</p>", unsafe_allow_html=True)

if "resume_text" not in st.session_state or not st.session_state.resume_text:
    st.warning(" Please upload a resume in the Candidate Portal first to see job matches.")
else:
    ats = int((st.session_state.ats_ml_score or {"score": 0})["score"])
    st.markdown(f"""
    <div class='header-card'>
        <div>
            <h3 style='margin:0; color:#0369A1; font-family:"Outfit", sans-serif;'>Your Resume Health Score:</h3>
            <p style='margin:4px 0 0 0; color:#0C4A6E; font-size: 14px;'>This dictates your competitiveness for the roles below.</p>
        </div>
        <h1 style='margin:0; font-family:"Outfit", sans-serif; font-size: 2.5rem; color:{"#10B981" if ats >= 70 else "#F59E0B" if ats >= 40 else "#EF4444"};'>{ats}/100</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Top Matches")
    
    matches = recommend_jobs(st.session_state.resume_text, top_n=15)
    
    if not matches:
        st.info("No matching jobs found in the database.")
    else:
        for job in matches:
            matched_html = "".join([f"<span style='background:#D1FAE5; color:#065F46; padding:2px 8px; border-radius:12px; font-size:12px; margin-right:4px;'>{s.capitalize()}</span>" for s in job.get('matched_skills', [])])
            missing_html = "".join([f"<span style='background:#FEE2E2; color:#991B1B; padding:2px 8px; border-radius:12px; font-size:12px; margin-right:4px;'>{s.capitalize()}</span>" for s in job.get('missing_skills', [])])
            
            explainability_html = ""
            if matched_html or missing_html:
                explainability_html = "<div style='margin-bottom: 10px;'>"
                if matched_html:
                    explainability_html += f"<div style='margin-bottom: 4px;'><b> Matched Skills:</b> {matched_html}</div>"
                if missing_html:
                    explainability_html += f"<div><b> Missing Skills:</b> {missing_html}</div>"
                explainability_html += "</div>"
                
            html_content = (
                f"<div class='job-card'>"
                f"<div style='display: flex; gap: 16px; margin-bottom: 16px;'>"
                f"  <div style='width: 60px; height: 60px; border-radius: 12px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); display: flex; align-items: center; justify-content: center; font-size: 24px; color: #A1A1AA;'></div>"
                f"  <div style='flex-grow: 1;'>"
                f"      <div class='match-score'>{job['match_score']}% Match <span style='font-size: 14px; font-weight: normal; color: #A1A1AA;'>({job.get('match_label', 'Good')})</span></div>"
                f"      <h3 class='job-title'>{job['title']}</h3>"
                f"      <h5 class='job-meta'>{job['company']} • {job['location']} • {job['salary']}</h5>"
                f"  </div>"
                f"</div>"
                f"<p style='margin: 0 0 10px 0; font-size: 12.5px; font-weight: bold; color: #22C55E;'> P95 Scaled • Top {100 - job.get('percentile', 0)}% Match in Corpus</p>"
                f"{explainability_html}"
                f"<input type='checkbox' id='desc_trigger_{job['id']}' class='desc-trigger'>"
                f"<div class='desc-wrapper'>"
                f"<p class='job-desc'>{job['description']}</p>"
                f"</div>"
                f"<label for='desc_trigger_{job['id']}' class='desc-btn'></label>"
                f"</div>"
            )
            st.markdown(html_content, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                # Generate Cover Letter
                if st.button(f" Cover Letter", key=f"cl_{job['id']}", type="secondary"):
                    with st.spinner("Generating PDF..."):
                        output_path = os.path.join(tempfile.gettempdir(), f"Cover_Letter_{job['company']}.pdf")
                        generate_cover_letter_pdf(
                            st.session_state.resume_text,
                            st.session_state.skills or [],
                            job['title'],
                            job['company'],
                            output_path
                        )
                        with open(output_path, "rb") as f:
                            st.download_button(
                                " Download PDF",
                                data=f,
                                file_name=f"Cover_Letter_{job['company']}.pdf",
                                mime="application/pdf",
                                key=f"dl_{job['id']}"
                            )
 
            with col2:
                # Mailto cold email
                skills_str = ", ".join(st.session_state.skills[:3]) if st.session_state.skills else "software development"
                subject = f"Application for {job['title']} - [Your Name]"
                body = f"Hi Hiring Team at {job['company']},\n\nI hope this email finds you well. I'm reaching out to express my strong interest in the {job['title']} position.\n\nWith my background in {skills_str}, I believe my technical skills and proactive approach would make me a great fit for this role.\n\nI have attached my resume for your consideration. I'd love to chat briefly if you have a few minutes next week.\n\nBest regards,\n[Your Name]"
                
                import urllib.parse
                mail_link = f"mailto:hr@{job['company'].lower().replace(' ', '')}.com?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
                st.markdown(f"<a href='{mail_link}' target='_blank' style='text-decoration: none;'><button style='width: 100%; padding: 8px 16px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.15); background: transparent; color: #F4F4F5; font-family: \"Plus Jakarta Sans\", sans-serif; font-weight: 600; cursor: pointer; transition: all 0.2s ease;'> Draft Email</button></a>", unsafe_allow_html=True)
            
            with col3:
                # Track in Database
                if st.button(" Track Application", key=f"track_{job['id']}", type="secondary"):
                    from database import insert_application
                    insert_application(job['company'], job['title'], "Wishlist", "")
                    st.toast(f"Added {job['company']} to your Application Tracker!")

import streamlit as st
import json
import os
import tempfile
import re
import ast



from ui_utils import inject_custom_css
inject_custom_css()

if not st.session_state.get("resume_text"):
    st.warning(" No resume loaded! Please upload your resume on the Dashboard first.")
    st.stop()

# ── Initialize State ──
if not st.session_state.get("edit_state_initialized"):
    from llm_utils import extract_resume_fields_via_llm
    from resume_builder import _extract_name, _extract_section
    
    with st.spinner(" Parsing and structuring resume details using AI..."):
        extracted_fields = extract_resume_fields_via_llm(st.session_state.resume_text)
        
    if extracted_fields:
        st.session_state.edit_name = extracted_fields.get("name", "")
        st.session_state.edit_summary = extracted_fields.get("summary", "")
        st.session_state.edit_skills_languages = extracted_fields.get("skills_languages", "")
        st.session_state.edit_skills_tools = extracted_fields.get("skills_tools", "")
        st.session_state.edit_skills_soft = extracted_fields.get("skills_soft", "")
        st.session_state.edit_experience = extracted_fields.get("experience", "")
        st.session_state.edit_education = extracted_fields.get("education", "")
        st.session_state.edit_achievements = extracted_fields.get("achievements", "")
        st.session_state.edit_certs_projects = extracted_fields.get("certs_projects", "")
    else:
        st.error(" **AI Extraction Failed!** Could not connect to Groq Llama-3. Please ensure your `GROQ_API_KEY` is set in your `.env` file and that you haven't hit rate limits. Falling back to unformatted raw text...")
        # Fallback to naive heuristics
        st.session_state.edit_name = _extract_name(st.session_state.resume_text)
        summary_text = ""
        paragraphs = [p.strip() for p in st.session_state.resume_text.split("\n\n") if p.strip()]
        if paragraphs:
            summary_text = paragraphs[0] if len(paragraphs[0].split()) > 5 else (paragraphs[1] if len(paragraphs) > 1 else "")
        st.session_state.edit_summary = summary_text
        st.session_state.edit_skills_languages = ""
        st.session_state.edit_skills_tools = ", ".join(st.session_state.skills) if st.session_state.skills else _extract_section(st.session_state.resume_text, "Skills")
        st.session_state.edit_skills_soft = ""
        st.session_state.edit_experience = _extract_section(st.session_state.resume_text, "Experience")
        st.session_state.edit_education = _extract_section(st.session_state.resume_text, "Education")
        st.session_state.edit_achievements = _extract_section(st.session_state.resume_text, "Achievements")
        certs = _extract_section(st.session_state.resume_text, "Certifications")
        projects = _extract_section(st.session_state.resume_text, "Projects")
        st.session_state.edit_certs_projects = "\n".join(filter(None, [certs, projects]))
        
    # Clear the old structured list states so they get re-initialized on every new resume parsing
    for list_key in [
        "edit_experience_list", "edit_education_list", "edit_projects_list",
        "edit_achievements_list", "edit_skills_languages_list",
        "edit_skills_tools_list", "edit_skills_soft_list"
    ]:
        if list_key in st.session_state:
            del st.session_state[list_key]
        
    st.session_state.edit_state_initialized = True
        


def sanitize_text_field(val, formatter):
    if not val:
        return val
    
    import ast
    import json
    
    # If the value is not a string, format it directly using the formatter
    if not isinstance(val, str):
        try:
            return formatter(val)
        except Exception:
            return str(val)
            
    val_stripped = val.strip()
    
    # Check if the entire field is a python list/dict literal
    if (val_stripped.startswith("[") and val_stripped.endswith("]")) or (val_stripped.startswith("{") and val_stripped.endswith("}")):
        try:
            parsed = ast.literal_eval(val_stripped)
            if parsed:
                return formatter(parsed)
        except Exception:
            try:
                parsed = json.loads(val_stripped)
                if parsed:
                    return formatter(parsed)
            except Exception:
                pass
                
    # Otherwise, clean line-by-line (e.g. if a single bullet line is a bracketed list)
    lines = val.split("\n")
    new_lines = []
    for line in lines:
        stripped = line.strip()
        bullet_marker = ""
        content = stripped
        
        if stripped.startswith("-"):
            bullet_marker = "- "
            content = stripped[1:].strip()
        elif stripped.startswith("•"):
            bullet_marker = "• "
            content = stripped[1:].strip()
        elif stripped.startswith("*"):
            bullet_marker = "* "
            content = stripped[1:].strip()
            
        if content.startswith("[") and content.endswith("]"):
            try:
                parsed = ast.literal_eval(content)
                if isinstance(parsed, list):
                    for item in parsed:
                        new_lines.append(f"{bullet_marker or '- '}{str(item).strip()}")
                    continue
            except Exception:
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        for item in parsed:
                            new_lines.append(f"{bullet_marker or '- '}{str(item).strip()}")
                        continue
                except Exception:
                    pass
        new_lines.append(line)
        
    return "\n".join(new_lines)

def sanitize_edit_states():
    from llm_utils import (
        _format_skills_to_text,
        _format_experience_to_text,
        _format_education_to_text,
        _format_projects_to_text,
        _format_achievements_to_text
    )
    
    mapping = {
        "edit_skills_languages": _format_skills_to_text,
        "edit_skills_tools": _format_skills_to_text,
        "edit_skills_soft": _format_skills_to_text,
        "edit_experience": _format_experience_to_text,
        "edit_education": _format_education_to_text,
        "edit_certs_projects": _format_projects_to_text,
        "edit_achievements": _format_achievements_to_text
    }
    
    for key, formatter in mapping.items():
        val = st.session_state.get(key)
        if val:
            sanitized = sanitize_text_field(val, formatter)
            if sanitized != val:
                st.session_state[key] = sanitized

# Run proactive sanitization on every script run
sanitize_edit_states()

def init_structured_lists():
    from resume_builder import _parse_resume_section
    
    # Experience
    if "edit_experience_list" not in st.session_state or st.session_state.edit_experience_list is None:
        if st.session_state.edit_experience:
            st.session_state.edit_experience_list = _parse_resume_section(st.session_state.edit_experience)
        else:
            st.session_state.edit_experience_list = []
            
    # Education
    if "edit_education_list" not in st.session_state or st.session_state.edit_education_list is None:
        if st.session_state.edit_education:
            parsed_edu = _parse_resume_section(st.session_state.edit_education)
            for entry in parsed_edu:
                entry["grades"] = []
                remaining_bullets = []
                for bullet in entry.get("bullets", []):
                    bullet_lower = bullet.lower()
                    if "grade" in bullet_lower or "cgpa" in bullet_lower or "gpa" in bullet_lower or "percentage" in bullet_lower or "class 10" in bullet_lower or "class 12" in bullet_lower:
                        grade_type = "CGPA"
                        if "class 10" in bullet_lower:
                            grade_type = "Class 10 %"
                        elif "class 12" in bullet_lower:
                            grade_type = "Class 12 %"
                        elif "gpa" in bullet_lower:
                            grade_type = "GPA"
                            
                        grade_val = bullet.split(":")[-1].strip() if ":" in bullet else bullet
                        entry["grades"].append({"type": grade_type, "value": grade_val})
                    else:
                        remaining_bullets.append(bullet)
                entry["bullets"] = remaining_bullets
            st.session_state.edit_education_list = parsed_edu
        else:
            st.session_state.edit_education_list = []
            
    # Projects
    if "edit_projects_list" not in st.session_state or st.session_state.edit_projects_list is None:
        if st.session_state.edit_certs_projects:
            st.session_state.edit_projects_list = _parse_resume_section(st.session_state.edit_certs_projects)
        else:
            st.session_state.edit_projects_list = []
            
    # Achievements
    if "edit_achievements_list" not in st.session_state or st.session_state.edit_achievements_list is None:
        if st.session_state.edit_achievements:
            ach_parsed = _parse_resume_section(st.session_state.edit_achievements)
            bullets = []
            for entry in ach_parsed:
                bullets.extend(entry.get("bullets", []))
            st.session_state.edit_achievements_list = bullets
        else:
            st.session_state.edit_achievements_list = []
            
    # Skills
    if "edit_skills_languages_list" not in st.session_state or st.session_state.edit_skills_languages_list is None:
        st.session_state.edit_skills_languages_list = [s.strip() for s in (st.session_state.edit_skills_languages or "").split(",") if s.strip()]
        
    if "edit_skills_tools_list" not in st.session_state or st.session_state.edit_skills_tools_list is None:
        st.session_state.edit_skills_tools_list = [s.strip() for s in (st.session_state.edit_skills_tools or "").split(",") if s.strip()]
        
    if "edit_skills_soft_list" not in st.session_state or st.session_state.edit_skills_soft_list is None:
        st.session_state.edit_skills_soft_list = [s.strip() for s in (st.session_state.edit_skills_soft or "").split(",") if s.strip()]

def sync_lists_to_text():
    from llm_utils import (
        _format_experience_to_text,
        _format_projects_to_text
    )
    
    if "edit_experience_list" in st.session_state and st.session_state.edit_experience_list is not None:
        st.session_state.edit_experience = _format_experience_to_text(st.session_state.edit_experience_list)
        
    if "edit_education_list" in st.session_state and st.session_state.edit_education_list is not None:
        edu_entries_formatted = []
        for entry in st.session_state.edit_education_list:
            bullets = list(entry.get("bullets", []))
            for g in entry.get("grades", []):
                bullets.append(f"Grade ({g['type']}): {g['value']}")
            copy_entry = dict(entry)
            copy_entry["bullets"] = bullets
            edu_entries_formatted.append(copy_entry)
            
        from llm_utils import _format_education_to_text
        st.session_state.edit_education = _format_education_to_text(edu_entries_formatted)
        
    if "edit_projects_list" in st.session_state and st.session_state.edit_projects_list is not None:
        st.session_state.edit_certs_projects = _format_projects_to_text(st.session_state.edit_projects_list)
        
    if "edit_achievements_list" in st.session_state and st.session_state.edit_achievements_list is not None:
        st.session_state.edit_achievements = "\n".join(f"- {a}" for a in st.session_state.edit_achievements_list if a)
        
    if "edit_skills_languages_list" in st.session_state and st.session_state.edit_skills_languages_list is not None:
        st.session_state.edit_skills_languages = ", ".join(st.session_state.edit_skills_languages_list)
    if "edit_skills_tools_list" in st.session_state and st.session_state.edit_skills_tools_list is not None:
        st.session_state.edit_skills_tools = ", ".join(st.session_state.edit_skills_tools_list)
    if "edit_skills_soft_list" in st.session_state and st.session_state.edit_skills_soft_list is not None:
        st.session_state.edit_skills_soft = ", ".join(st.session_state.edit_skills_soft_list)

def enhance_single_project(project_entry):
    from llm_utils import call_groq_api
    import json
    bullets_str = "\n".join(f"- {b}" for b in project_entry.get("bullets", []))
    prompt = f"""You are a professional resume writer and ATS optimization expert. Enhance this specific project entry to be highly professional and ATS-optimized.
Use active verbs and quantify results where possible. Follow the Google X-Y-Z formula for the description bullets.

PROJECT DETAILS:
Title/Heading: {project_entry.get('title', '')}
Subtitle/Role: {project_entry.get('subtitle', '')}
Current Description Bullets:
{bullets_str}

Return ONLY a JSON object in this format:
{{
  "title": "Enhanced Title",
  "subtitle": "Enhanced Subtitle/Role",
  "bullets": [
    "Enhanced bullet 1",
    "Enhanced bullet 2"
  ]
}}
Do NOT output any markdown code blocks (```json or ```) or explanations outside the JSON. Respond with the raw JSON string only.
"""
    system_prompt = "You are a professional resume writer. Respond with raw JSON only."
    try:
        response_text = call_groq_api(prompt, system_prompt)
        if response_text.startswith("```"):
            response_text = re.sub(r"^```(?:json)?\n", "", response_text)
            response_text = re.sub(r"\n```$", "", response_text)
            response_text = response_text.strip()
        data = json.loads(response_text)
        return data
    except Exception as e:
        st.error(f"Error enhancing project: {e}")
        return None

def enhance_single_achievement(achievement_text):
    from llm_utils import call_groq_api
    import json
    prompt = f"""You are a professional resume writer and ATS optimization expert. Enhance this single achievement statement to be highly professional, impactful, and clear.
Highlight honors, academic standing, or competition rankings. Keep it to a single concise sentence.

ACHIEVEMENT STATEMENT:
{achievement_text}

Return ONLY a JSON object in this format:
{{
  "enhanced_text": "Enhanced achievement statement"
}}
Do NOT output any markdown code blocks (```json or ```) or explanations outside the JSON. Respond with the raw JSON string only.
"""
    system_prompt = "You are a professional resume writer. Respond with raw JSON only."
    try:
        response_text = call_groq_api(prompt, system_prompt)
        if response_text.startswith("```"):
            response_text = re.sub(r"^```(?:json)?\n", "", response_text)
            response_text = re.sub(r"\n```$", "", response_text)
            response_text = response_text.strip()
        data = json.loads(response_text)
        return data.get("enhanced_text", achievement_text)
    except Exception as e:
        st.error(f"Error enhancing achievement: {e}")
        return None

def render_resume_editor():
    import json
    import base64
    st.header(" Interactive Resume Editor & AI Templates")
    st.markdown("Modify your resume details below. Any edits or styling changes will automatically update the PDF preview on the right.")
    
    role = st.session_state.predicted_role or "Professional"
    
    # Initialize structured lists from plain text area states
    init_structured_lists()
    
    # Helper to render PDF pages as styled images to bypass Chrome security blocks and maintain proportions
    def display_pdf_as_images(pdf_path):
        import fitz
        import base64
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                base64_img = base64.b64encode(img_bytes).decode('utf-8')
                st.markdown(
                    f'<div style="display: flex; justify-content: center; margin-bottom: 20px;">'
                    f'<img src="data:image/png;base64,{base64_img}" style="width: 100%; max-width: 600px; border: 1px solid #cbd5e1; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1);" />'
                    f'</div>',
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Could not render PDF preview: {e}")

    # Divide screen into Edit Controls (Left) and Live PDF Preview (Right)
    col_edit, col_preview = st.columns([1.1, 0.9])
    
    with col_edit:
        st.subheader(" Edit Content")
        
        col_name, col_role = st.columns(2)
        with col_name:
            st.session_state.edit_name = st.text_input("Full Name:", value=st.session_state.edit_name or "")
        with col_role:
            st.session_state.predicted_role = st.text_input("Target Role / Title:", value=role)
            
        st.session_state.edit_summary = st.text_area("Professional Summary:", value=st.session_state.edit_summary or "", height=100)
        
        # 1. Expander: Skills
        with st.expander(" Skills", expanded=True):
            st.markdown("**Languages**")
            for idx, lang in enumerate(list(st.session_state.edit_skills_languages_list)):
                col_val, col_del = st.columns([0.85, 0.15])
                with col_val:
                    st.session_state.edit_skills_languages_list[idx] = st.text_input(f"Language #{idx+1}", value=lang, key=f"lang_{idx}", label_visibility="collapsed")
                with col_del:
                    if st.button("", key=f"del_lang_{idx}"):
                        st.session_state.edit_skills_languages_list.pop(idx)
                        st.rerun()
            if st.button(" Add Language", key="add_lang"):
                st.session_state.edit_skills_languages_list.append("")
                st.rerun()
                
            st.markdown("**Tools & Technologies**")
            for idx, tool in enumerate(list(st.session_state.edit_skills_tools_list)):
                col_val, col_del = st.columns([0.85, 0.15])
                with col_val:
                    st.session_state.edit_skills_tools_list[idx] = st.text_input(f"Tool #{idx+1}", value=tool, key=f"tool_{idx}", label_visibility="collapsed")
                with col_del:
                    if st.button("", key=f"del_tool_{idx}"):
                        st.session_state.edit_skills_tools_list.pop(idx)
                        st.rerun()
            if st.button(" Add Tool / Tech", key="add_tool"):
                st.session_state.edit_skills_tools_list.append("")
                st.rerun()
                
            st.markdown("**Soft Skills**")
            for idx, soft in enumerate(list(st.session_state.edit_skills_soft_list)):
                col_val, col_del = st.columns([0.85, 0.15])
                with col_val:
                    st.session_state.edit_skills_soft_list[idx] = st.text_input(f"Soft Skill #{idx+1}", value=soft, key=f"soft_{idx}", label_visibility="collapsed")
                with col_del:
                    if st.button("", key=f"del_soft_{idx}"):
                        st.session_state.edit_skills_soft_list.pop(idx)
                        st.rerun()
            if st.button(" Add Soft Skill", key="add_soft"):
                st.session_state.edit_skills_soft_list.append("")
                st.rerun()

        # 2. Expander: Work Experience
        with st.expander(" Work Experience", expanded=False):
            for idx, entry in enumerate(list(st.session_state.edit_experience_list)):
                st.markdown(f"---")
                col_title, col_del = st.columns([0.85, 0.15])
                with col_title:
                    st.markdown(f"**Role #{idx+1}: {entry.get('title') or 'New Role'}**")
                with col_del:
                    if st.button(" Delete", key=f"del_exp_{idx}"):
                        st.session_state.edit_experience_list.pop(idx)
                        st.rerun()
                        
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.edit_experience_list[idx]["title"] = st.text_input("Company / Organization:", value=entry.get("title", ""), key=f"exp_{idx}_title")
                    st.session_state.edit_experience_list[idx]["location"] = st.text_input("Location:", value=entry.get("location", ""), key=f"exp_{idx}_location")
                with col2:
                    st.session_state.edit_experience_list[idx]["subtitle"] = st.text_input("Position / Title:", value=entry.get("subtitle", ""), key=f"exp_{idx}_subtitle")
                    st.session_state.edit_experience_list[idx]["dates"] = st.text_input("Dates / Duration:", value=entry.get("dates", ""), key=f"exp_{idx}_dates")
                    
                bullets_text = "\n".join(entry.get("bullets", []))
                new_bullets_text = st.text_area("Description (One bullet point per line):", value=bullets_text, key=f"exp_{idx}_bullets", height=120)
                st.session_state.edit_experience_list[idx]["bullets"] = [b.strip() for b in new_bullets_text.split("\n") if b.strip()]
                
            if st.button(" Add Work Experience", key="add_exp"):
                st.session_state.edit_experience_list.append({
                    "title": "",
                    "subtitle": "",
                    "location": "",
                    "dates": "",
                    "bullets": [""]
                })
                st.rerun()

        # 3. Expander: Education
        with st.expander(" Education", expanded=False):
            for idx, entry in enumerate(list(st.session_state.edit_education_list)):
                st.markdown(f"---")
                col_title, col_del = st.columns([0.85, 0.15])
                with col_title:
                    st.markdown(f"**Education #{idx+1}: {entry.get('title') or 'New Institution'}**")
                with col_del:
                    if st.button(" Delete", key=f"del_edu_{idx}"):
                        st.session_state.edit_education_list.pop(idx)
                        st.rerun()
                        
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.edit_education_list[idx]["title"] = st.text_input("Institution / School / College:", value=entry.get("title", ""), key=f"edu_{idx}_title")
                    st.session_state.edit_education_list[idx]["location"] = st.text_input("Location:", value=entry.get("location", ""), key=f"edu_{idx}_location")
                with col2:
                    st.session_state.edit_education_list[idx]["subtitle"] = st.text_input("Degree / Course:", value=entry.get("subtitle", ""), key=f"edu_{idx}_subtitle")
                    st.session_state.edit_education_list[idx]["dates"] = st.text_input("Dates / Years (e.g. 2018 - 2022):", value=entry.get("dates", ""), key=f"edu_{idx}_dates")
                    
                st.markdown("**Grades & Academic Standing:**")
                grades = entry.get("grades", [])
                for g_idx, g in enumerate(list(grades)):
                    col_type, col_val, col_gdel = st.columns([0.5, 0.4, 0.1])
                    with col_type:
                        grade_types = ["CGPA", "Class 10 %", "Class 12 %", "GPA", "Percentage"]
                        default_type_idx = grade_types.index(g["type"]) if g["type"] in grade_types else 0
                        selected_type = st.selectbox("Grade Type:", grade_types, index=default_type_idx, key=f"edu_{idx}_gtype_{g_idx}")
                        st.session_state.edit_education_list[idx]["grades"][g_idx]["type"] = selected_type
                    with col_val:
                        st.session_state.edit_education_list[idx]["grades"][g_idx]["value"] = st.text_input("Value / Score:", value=g["value"], key=f"edu_{idx}_gval_{g_idx}")
                    with col_gdel:
                        st.write("")
                        if st.button("", key=f"edu_{idx}_gdel_{g_idx}"):
                            st.session_state.edit_education_list[idx]["grades"].pop(g_idx)
                            st.rerun()
                if st.button(" Add Grade / Score", key=f"edu_{idx}_gadd"):
                    if "grades" not in st.session_state.edit_education_list[idx]:
                        st.session_state.edit_education_list[idx]["grades"] = []
                    st.session_state.edit_education_list[idx]["grades"].append({"type": "CGPA", "value": ""})
                    st.rerun()
                    
                bullets_text = "\n".join(entry.get("bullets", []))
                new_bullets_text = st.text_area("Other details/honors (One per line):", value=bullets_text, key=f"edu_{idx}_bullets", height=80)
                st.session_state.edit_education_list[idx]["bullets"] = [b.strip() for b in new_bullets_text.split("\n") if b.strip()]
                
            if st.button(" Add Education", key="add_edu"):
                st.session_state.edit_education_list.append({
                    "title": "",
                    "subtitle": "",
                    "location": "",
                    "dates": "",
                    "grades": [],
                    "bullets": []
                })
                st.rerun()

        # 4. Expander: Projects & Certifications
        with st.expander(" Certifications & Projects", expanded=False):
            for idx, entry in enumerate(list(st.session_state.edit_projects_list)):
                st.markdown(f"---")
                col_title, col_del = st.columns([0.85, 0.15])
                with col_title:
                    st.markdown(f"**Project #{idx+1}: {entry.get('title') or 'New Project'}**")
                with col_del:
                    if st.button(" Delete", key=f"del_proj_{idx}"):
                        st.session_state.edit_projects_list.pop(idx)
                        st.rerun()
                        
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.edit_projects_list[idx]["title"] = st.text_input("Project Title:", value=entry.get("title", ""), key=f"proj_{idx}_title")
                    st.session_state.edit_projects_list[idx]["location"] = st.text_input("Link / URL:", value=entry.get("location", ""), key=f"proj_{idx}_location")
                with col2:
                    st.session_state.edit_projects_list[idx]["subtitle"] = st.text_input("Role / Technologies:", value=entry.get("subtitle", ""), key=f"proj_{idx}_subtitle")
                    st.session_state.edit_projects_list[idx]["dates"] = st.text_input("Dates / Duration:", value=entry.get("dates", ""), key=f"proj_{idx}_dates")
                    
                bullets_text = "\n".join(entry.get("bullets", []))
                new_bullets_text = st.text_area("Project Description Bullets (One per line):", value=bullets_text, key=f"proj_{idx}_bullets", height=120)
                st.session_state.edit_projects_list[idx]["bullets"] = [b.strip() for b in new_bullets_text.split("\n") if b.strip()]
                
                # Single Project AI Enhancement Button
                if st.button(" Enhance this Project with AI", key=f"proj_{idx}_ai"):
                    with st.spinner("Optimizing project using Groq AI..."):
                        enhanced = enhance_single_project(st.session_state.edit_projects_list[idx])
                        if enhanced:
                            st.session_state.edit_projects_list[idx]["title"] = enhanced.get("title", st.session_state.edit_projects_list[idx]["title"])
                            st.session_state.edit_projects_list[idx]["subtitle"] = enhanced.get("subtitle", st.session_state.edit_projects_list[idx]["subtitle"])
                            st.session_state.edit_projects_list[idx]["bullets"] = enhanced.get("bullets", st.session_state.edit_projects_list[idx]["bullets"])
                            st.success(" Project successfully enhanced!")
                            st.rerun()
                            
            if st.button(" Add Project", key="add_proj"):
                st.session_state.edit_projects_list.append({
                    "title": "",
                    "subtitle": "",
                    "location": "",
                    "dates": "",
                    "bullets": [""]
                })
                st.rerun()

        # 5. Expander: Achievements
        with st.expander(" Achievements", expanded=False):
            for idx, bullet in enumerate(list(st.session_state.edit_achievements_list)):
                col_val, col_ai, col_del = st.columns([0.7, 0.2, 0.1])
                with col_val:
                    st.session_state.edit_achievements_list[idx] = st.text_input(f"Achievement #{idx+1}", value=bullet, key=f"ach_{idx}", label_visibility="collapsed")
                with col_ai:
                    if st.button(" Enhance", key=f"ach_{idx}_ai"):
                        with st.spinner("Enhancing achievement..."):
                            enhanced = enhance_single_achievement(st.session_state.edit_achievements_list[idx])
                            if enhanced:
                                st.session_state.edit_achievements_list[idx] = enhanced
                                st.success(" Enhanced!")
                                st.rerun()
                with col_del:
                    if st.button("", key=f"ach_{idx}_del"):
                        st.session_state.edit_achievements_list.pop(idx)
                        st.rerun()
            if st.button(" Add Achievement", key="add_ach"):
                st.session_state.edit_achievements_list.append("")
                st.rerun()

        # AI Enhance Section
        st.markdown("---")
        st.subheader(" AI Resume Enhancement")
        
        st.markdown("Select which sections you want the AI to enhance:")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            enhance_summary = st.checkbox("Professional Summary", value=True, key="opt_summary")
            enhance_skills = st.checkbox("Skills (Languages, Tools, Soft Skills)", value=True, key="opt_skills")
            enhance_experience = st.checkbox("Work Experience", value=True, key="opt_experience")
        with col_opt2:
            enhance_education = st.checkbox("Education", value=True, key="opt_education")
            enhance_projects = st.checkbox("Certifications & Projects", value=True, key="opt_projects")
            enhance_achievements = st.checkbox("Achievements", value=True, key="opt_achievements")
            
        selected_sections = []
        if enhance_summary: selected_sections.append("summary")
        if enhance_skills: selected_sections.extend(["skills_languages", "skills_tools", "skills_soft"])
        if enhance_experience: selected_sections.append("experience")
        if enhance_education: selected_sections.append("education")
        if enhance_projects: selected_sections.append("certs_projects")
        if enhance_achievements: selected_sections.append("achievements")

        st.markdown(" **Optional Metrics Context** (Provide answers below to help the AI add quantifiable outcomes):")
        q_metrics = st.text_input("1. Metrics & Outcomes: Did you improve speed, reduce cost, or increase efficiency? (e.g. 'Reduced latency by 35%')", value="", key="q_metrics")
        q_scale = st.text_input("2. Scale & Volume: How many users, servers, or data volume did you work with? (e.g. '15k daily active users')", value="", key="q_scale")
        q_tech = st.text_input("3. Tech Stack Detail: What specific languages, tools, or libraries did you use? (e.g. 'Docker, FastAPI, React')", value="", key="q_tech")
        q_ownership = st.text_input("4. Ownership & Scope: What was your specific contribution or team size? (e.g. 'Led a team of 4 to design database schema')", value="", key="q_ownership")

        if st.button(" Enhance with Groq AI", type="secondary", use_container_width=True):
            if not selected_sections:
                st.warning(" Please select at least one section to enhance!")
            else:
                with st.spinner("Analyzing feedback & generating improvements with Groq Llama-3..."):
                    # Gather weak bullets and feedback points
                    feedback_bullets = []
                    if st.session_state.bullet_results:
                        for b in st.session_state.bullet_results:
                            if b.get("label") == "Weak":
                                feedback_bullets.append(f"- Bullet: \"{b['text']}\" -> Feedback: missing metrics or strong verbs.")
                    
                    feedback_issues = []
                    if st.session_state.issues:
                        for issue in st.session_state.issues:
                            feedback_issues.append(f"- Formatting Issue: {issue.get('issue')}")
                    
                    feedback_str = "\n".join(feedback_bullets + feedback_issues)
                    
                    sections_to_enhance_str = ", ".join(selected_sections)
                    
                    additional_context_parts = []
                    if q_metrics.strip(): additional_context_parts.append(f"- Metrics/Outcomes: {q_metrics.strip()}")
                    if q_scale.strip(): additional_context_parts.append(f"- Scale/Volume: {q_scale.strip()}")
                    if q_tech.strip(): additional_context_parts.append(f"- Tech Stack: {q_tech.strip()}")
                    if q_ownership.strip(): additional_context_parts.append(f"- Ownership/Scope: {q_ownership.strip()}")
                    additional_context_str = "\n".join(additional_context_parts) if additional_context_parts else "None"
                    
                    prompt = f"""You are a professional resume writer and ATS optimization expert. Your task is to enhance the selected sections of the candidate's resume to make them highly impactful, professional, and ATS-friendly.

CRITICAL TRUTHFULNESS RULES - NEVER VIOLATE:
1. DO NOT add any skill, tool, technology, or certification that is not explicitly mentioned in the original resume.
2. DO NOT invent numeric achievements (e.g., "increased sales by 30%") unless they exist in the original text. You may rephrase existing metrics to be more impact-driven, but never fabricate new numbers.
3. DO NOT add company names, product names, or technical terms not supported by the candidate's actual experience.
4. DO NOT upgrade experience level (e.g., "Junior Software Engineer" -> "Senior Software Engineer").
5. DO NOT extend employment dates or change timelines. Copy date ranges exactly as they appear, including months.
6. Preserve factual accuracy - only use information provided by the candidate.
7. NEVER remove existing skills, certifications, languages, or awards. You may reorder by relevance, but every original item must remain.

SECTIONS TO ENHANCE:
{sections_to_enhance_str}

FEEDBACK AND ISSUES TO RESOLVE (if any):
{feedback_str}

CANDIDATE'S ADDITIONAL CONTEXT (incorporate this context to add quantifiable metrics and impact):
{additional_context_str}

ORIGINAL RESUME DATA:
- Name: {st.session_state.edit_name}
- Target Role: {st.session_state.predicted_role}
- Summary: {st.session_state.edit_summary}
- Skills (Languages): {st.session_state.edit_skills_languages}
- Skills (Tools & Tech): {st.session_state.edit_skills_tools}
- Skills (Soft Skills): {st.session_state.edit_skills_soft}
- Work Experience: {st.session_state.edit_experience}
- Education: {st.session_state.edit_education}
- Certifications & Projects: {st.session_state.edit_certs_projects}
- Achievements: {st.session_state.edit_achievements}

ENHANCEMENT INSTRUCTIONS FOR SECTIONS:
- summary: Rewrite to be a concise, powerful professional summary (2-3 sentences) emphasizing the candidate's core expertise and target role. Do not use generic buzzwords.
- skills_languages / skills_tools / skills_soft: Keep them clean, industry-standard, and well-organized.
- experience: Rewrite bullet points using the Google X-Y-Z formula (Accomplished [X] as measured by [Y], by doing [Z]). Start with strong action verbs. Highlight responsibilities, tools used, and outcomes. Keep all original dates, company names, and locations unchanged.
- education: Clean up formatting, preserve degrees, institutions, and years.
- certs_projects: Enhance project descriptions and bullets to clearly show technical implementation details and results.
- achievements: Rephrase achievements to highlight honors, academic performance, or competition rankings clearly.

OUTPUT FORMAT:
Return ONLY a valid JSON object containing all the keys listed below. For any section NOT selected for enhancement, you MUST copy the original value exactly as it is without any changes.

Expected JSON output format:
{{
  "name": "{st.session_state.edit_name}",
  "summary": "...",
  "skills_languages": "...",
  "skills_tools": "...",
  "skills_soft": "...",
  "experience": "...",
  "education": "...",
  "certs_projects": "...",
  "achievements": "..."
}}
Do NOT output any markdown code block backticks (```json or ```), styling, or explanations outside the JSON. Respond with the raw JSON string only.
"""
                    system_prompt = "You are a professional resume writer. Respond with raw JSON only. Do not include markdown code block backticks (```json or ```)."
                    
                    from llm_utils import call_groq_api
                    response_text = call_groq_api(prompt, system_prompt)
                    
                    # Clean response text in case markdown block wrapper is outputted anyway
                    if response_text.startswith("```"):
                        response_text = re.sub(r"^```(?:json)?\n", "", response_text)
                        response_text = re.sub(r"\n```$", "", response_text)
                        response_text = response_text.strip()
                        
                    try:
                        improved_data = json.loads(response_text)
                        from llm_utils import (
                            _format_skills_to_text,
                            _format_experience_to_text,
                            _format_education_to_text,
                            _format_projects_to_text,
                            _format_achievements_to_text
                        )
                        st.session_state.edit_name = str(improved_data.get("name", st.session_state.edit_name)).strip()
                        
                        if enhance_summary:
                            st.session_state.edit_summary = str(improved_data.get("summary", st.session_state.edit_summary)).strip()
                        if enhance_skills:
                            st.session_state.edit_skills_languages = _format_skills_to_text(improved_data.get("skills_languages", st.session_state.edit_skills_languages))
                            st.session_state.edit_skills_tools = _format_skills_to_text(improved_data.get("skills_tools", st.session_state.edit_skills_tools))
                            st.session_state.edit_skills_soft = _format_skills_to_text(improved_data.get("skills_soft", st.session_state.edit_skills_soft))
                        if enhance_experience:
                            st.session_state.edit_experience = _format_experience_to_text(improved_data.get("experience", st.session_state.edit_experience))
                        if enhance_education:
                            st.session_state.edit_education = _format_education_to_text(improved_data.get("education", st.session_state.edit_education))
                        if enhance_projects:
                            st.session_state.edit_certs_projects = _format_projects_to_text(improved_data.get("certs_projects", st.session_state.edit_certs_projects))
                        if enhance_achievements:
                            st.session_state.edit_achievements = _format_achievements_to_text(improved_data.get("achievements", st.session_state.edit_achievements))
                            
                        # Clear old lists so they refresh on the next run
                        for list_key in [
                            "edit_experience_list", "edit_education_list", "edit_projects_list",
                            "edit_achievements_list", "edit_skills_languages_list",
                            "edit_skills_tools_list", "edit_skills_soft_list"
                        ]:
                            if list_key in st.session_state:
                                del st.session_state[list_key]
                                
                        st.success(" Selected resume sections successfully enhanced by Groq AI! Review the changes below.")
                        st.rerun()
                    except Exception as e:
                        st.error("Failed to parse AI response. The response was:")
                        st.code(response_text)
                    
        # PDF Layout & Styling Panel
        st.markdown("---")
        st.subheader(" PDF Styling & Templates")
        with st.expander("PDF Formatting Options", expanded=True):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                pdf_template = st.selectbox("Template Layout Style", ["Clean", "Modern Two-Column", "Vivid", "Classic"], index=0)
                pdf_font = st.selectbox("Font Family", ["Helvetica", "Times", "Courier"], index=0)
                pdf_color = st.selectbox("Color Theme", ["Dark Blue", "Black", "Green", "Crimson Red", "Slate Grey"], index=0)
            with col_f2:
                pdf_page_size = st.selectbox("Page Size", ["A4", "Letter"], index=0)
                pdf_font_size = st.slider("Base Font Size (pt)", min_value=8, max_value=14, value=10, step=1)
                pdf_margin = st.slider("Page Margin (mm)", min_value=10, max_value=30, value=15, step=5)
                
        # Sync structured lists back to plain text state variables
        sync_lists_to_text()

    with col_preview:
        st.subheader(" Live PDF Preview")
        
        # Compile PDF on page load/interaction
        with st.spinner("Updating PDF preview..."):
            from resume_builder import generate_enhanced_pdf_direct
            output_path = os.path.join(tempfile.gettempdir(), "live_preview_styled_resume.pdf")
            try:
                generate_enhanced_pdf_direct(
                    name=st.session_state.edit_name,
                    summary=st.session_state.edit_summary,
                    skills_text="",
                    experience=st.session_state.edit_experience,
                    education=st.session_state.edit_education,
                    achievements=st.session_state.edit_achievements,
                    certs_projects=st.session_state.edit_certs_projects,
                    output_path=output_path,
                    font_family=pdf_font,
                    base_font_size=pdf_font_size,
                    margin=pdf_margin,
                    primary_color_name=pdf_color,
                    page_size=pdf_page_size,
                    template_style=pdf_template,
                    predicted_role=st.session_state.predicted_role,
                    original_text=st.session_state.resume_text,
                    skills_languages=st.session_state.edit_skills_languages,
                    skills_tools=st.session_state.edit_skills_tools,
                    skills_soft=st.session_state.edit_skills_soft
                )
                
                # Show PDF as images instead of an iframe to bypass Chrome security blocks
                display_pdf_as_images(output_path)
                
                # Download button below the preview
                st.markdown(" ")
                with open(output_path, "rb") as f:
                    st.download_button(" Download Styled PDF Resume", data=f.read(), file_name="enhanced_styled_resume.pdf", mime="application/pdf", use_container_width=True)
            except Exception as e:
                st.error(f"Failed to generate live preview: {e}")
                st.exception(e)


# Render the editor automatically
render_resume_editor()

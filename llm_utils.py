import os
import requests
import json
import re

def _parse_if_serialized(val):
    if not isinstance(val, str):
        return val
    stripped = val.strip()
    if (stripped.startswith("[") and stripped.endswith("]")) or (stripped.startswith("{") and stripped.endswith("}")):
        import ast
        import json
        try:
            return ast.literal_eval(stripped)
        except Exception:
            try:
                return json.loads(stripped)
            except Exception:
                pass
    return val

def _format_skills_to_text(skills) -> str:
    skills = _parse_if_serialized(skills)
    if isinstance(skills, str):
        return skills
    if isinstance(skills, list):
        return ", ".join(str(s) for s in skills if s)
    if isinstance(skills, dict):
        items = []
        for k, v in skills.items():
            if isinstance(v, list):
                items.extend(str(x) for x in v if x)
            elif v:
                items.append(str(v))
        return ", ".join(items)
    return str(skills) if skills else ""


def _format_experience_to_text(experience_entries) -> str:
    experience_entries = _parse_if_serialized(experience_entries)
    if isinstance(experience_entries, dict):
        experience_entries = [experience_entries]
    if isinstance(experience_entries, str):
        return experience_entries
    if not isinstance(experience_entries, list):
        return str(experience_entries) if experience_entries else ""
        
    text_blocks = []
    for entry in experience_entries:
        if not isinstance(entry, dict):
            text_blocks.append(str(entry))
            continue
        # Extract fields
        org = entry.get("organization") or entry.get("company") or entry.get("title") or ""
        role = entry.get("position") or entry.get("role") or entry.get("subtitle") or ""
        dur = entry.get("duration") or entry.get("years") or entry.get("dates") or ""
        loc = entry.get("location") or ""
        bullets = entry.get("bullet_points") or entry.get("bullets") or entry.get("description") or []
        
        # Build header line
        header = ""
        if org and role:
            header = f"{org} | {role}"
        elif org:
            header = org
        elif role:
            header = role
            
        # Build meta line
        meta = ""
        if loc and dur:
            meta = f"{loc} | {dur}"
        elif loc:
            meta = loc
        elif dur:
            meta = dur
            
        block = header
        if meta:
            block += f"\n{meta}"
            
        if isinstance(bullets, list):
            for b in bullets:
                block += f"\n- {b}"
        elif isinstance(bullets, str) and bullets.strip():
            for line in bullets.split("\n"):
                line = line.strip()
                if line:
                    if not (line.startswith("-") or line.startswith("•") or line.startswith("*")):
                        block += f"\n- {line}"
                    else:
                        block += f"\n{line}"
                        
        text_blocks.append(block)
        
    return "\n\n".join(text_blocks)


def _format_education_to_text(education_entries) -> str:
    education_entries = _parse_if_serialized(education_entries)
    if isinstance(education_entries, dict):
        education_entries = [education_entries]
    if isinstance(education_entries, str):
        return education_entries
    if not isinstance(education_entries, list):
        return str(education_entries) if education_entries else ""
        
    text_blocks = []
    for entry in education_entries:
        if not isinstance(entry, dict):
            text_blocks.append(str(entry))
            continue
            
        inst = entry.get("institution") or entry.get("school") or entry.get("title") or ""
        deg = entry.get("degree") or entry.get("subtitle") or ""
        dur = entry.get("duration") or entry.get("years") or entry.get("dates") or ""
        loc = entry.get("location") or ""
        grade = entry.get("cgpa") or entry.get("grade") or entry.get("gpa") or ""
        
        header = ""
        if inst and deg:
            header = f"{inst} | {deg}"
        elif inst:
            header = inst
        elif deg:
            header = deg
            
        meta = ""
        if loc and dur:
            meta = f"{loc} | {dur}"
        elif loc:
            meta = loc
        elif dur:
            meta = dur
            
        block = header
        if meta:
            block += f"\n{meta}"
            
        if grade:
            grade_str = str(grade).strip()
            if grade_str.startswith("-") or grade_str.startswith("•") or grade_str.startswith("*"):
                block += f"\n{grade_str}"
            else:
                block += f"\n- GPA/Grade: {grade_str}"
            
        bullets = entry.get("bullet_points") or entry.get("bullets") or entry.get("description") or []
        if isinstance(bullets, list):
            for b in bullets:
                block += f"\n- {b}"
        elif isinstance(bullets, str) and bullets.strip():
            for line in bullets.split("\n"):
                line = line.strip()
                if line:
                    if not (line.startswith("-") or line.startswith("•") or line.startswith("*")):
                        block += f"\n- {line}"
                    else:
                        block += f"\n{line}"
                        
        text_blocks.append(block)
        
    return "\n\n".join(text_blocks)


def _format_projects_to_text(project_entries) -> str:
    project_entries = _parse_if_serialized(project_entries)
    if isinstance(project_entries, dict):
        project_entries = [project_entries]
    if isinstance(project_entries, str):
        return project_entries
    if not isinstance(project_entries, list):
        return str(project_entries) if project_entries else ""
        
    text_blocks = []
    for entry in project_entries:
        if not isinstance(entry, dict):
            text_blocks.append(str(entry))
            continue
            
        name = entry.get("name") or entry.get("title") or ""
        desc = entry.get("description") or ""
        url = entry.get("url") or entry.get("github") or entry.get("website") or ""
        bullets = entry.get("bullet_points") or entry.get("bullets") or []
        dur = entry.get("duration") or entry.get("years") or entry.get("dates") or ""
        role = entry.get("role") or entry.get("subtitle") or ""
        
        # If desc is actually a list, treat it as bullets
        if isinstance(desc, list):
            if not bullets:
                bullets = desc
            desc = ""
            
        header = ""
        if name and role:
            header = f"{name} | {role}"
        elif name:
            header = name
            
        meta = ""
        if url and dur:
            meta = f"{url} | {dur}"
        elif url:
            meta = url
        elif dur:
            meta = dur
            
        block = header
        if meta:
            block += f"\n{meta}"
            
        if isinstance(desc, str) and desc.strip():
            desc_str = desc.strip()
            if desc_str.startswith("-") or desc_str.startswith("•") or desc_str.startswith("*"):
                block += f"\n{desc_str}"
            else:
                block += f"\n- {desc_str}"
            
        if isinstance(bullets, list):
            for b in bullets:
                block += f"\n- {b}"
        elif isinstance(bullets, str) and bullets.strip():
            for line in bullets.split("\n"):
                line = line.strip()
                if line:
                    if not (line.startswith("-") or line.startswith("•") or line.startswith("*")):
                        block += f"\n- {line}"
                    else:
                        block += f"\n{line}"
                        
        text_blocks.append(block)
        
    return "\n\n".join(text_blocks)


def _format_achievements_to_text(achievement_entries) -> str:
    achievement_entries = _parse_if_serialized(achievement_entries)
    if isinstance(achievement_entries, dict):
        achievement_entries = [achievement_entries]
    if isinstance(achievement_entries, str):
        return achievement_entries
    if not isinstance(achievement_entries, list):
        return str(achievement_entries) if achievement_entries else ""
        
    bullets = []
    for entry in achievement_entries:
        if isinstance(entry, str):
            entry_stripped = entry.strip()
            if entry_stripped.startswith("-") or entry_stripped.startswith("•") or entry_stripped.startswith("*"):
                bullets.append(entry_stripped)
            else:
                bullets.append(f"- {entry_stripped}")
        elif isinstance(entry, dict):
            title = entry.get("title") or ""
            desc = entry.get("description") or ""
            if title and desc:
                bullets.append(f"- {title}: {desc}")
            elif title:
                bullets.append(f"- {title}")
            elif desc:
                bullets.append(f"- {desc}")
    return "\n".join(bullets)


def extract_json_from_text(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text
        
    brace_count = 0
    in_string = False
    escape = False
    
    for idx in range(start, len(text)):
        char = text[idx]
        
        if escape:
            escape = False
            continue
            
        if char == "\\":
            escape = True
            continue
            
        if char == '"':
            in_string = not in_string
            continue
            
        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start:idx+1]
                    
    return text[start:]

def extract_resume_fields_via_llm(resume_text: str) -> dict:
    """Use Groq LLM to parse raw resume text into structured fields.
    
    Returns a dictionary with name, summary, skills_languages, skills_tools, skills_soft,
    experience, education, achievements, and certs_projects.
    """
    prompt = f"""You are a professional resume parser. Extract the details of the candidate's resume from the raw text into structured JSON matching the schema below.

Categorize their skills specifically into:
- Languages (e.g., Python, SQL, C++, English, Spanish)
- Tools & Technologies (e.g., Docker, Kubernetes, Git, PyTorch, React, AWS)
- Soft Skills (e.g., Leadership, Project Management, Communication)

Extract all entries for experience, education, achievements, and certifications/projects completely. Do not summarize or truncate them. Keep all detailed bullet points.

Return ONLY a valid JSON object matching this schema exactly:
{{
  "name": "Candidate's Full Name",
  "summary": "Professional Summary",
  "skills_languages": ["Language 1", "Language 2"],
  "skills_tools": ["Tool 1", "Tool 2"],
  "skills_soft": ["Soft Skill 1", "Soft Skill 2"],
  "experience": [
    {{
      "organization": "Company or Organization Name",
      "position": "Job Title or Role",
      "location": "Location (City, Country/State)",
      "duration": "Dates/Duration (e.g., May 2025 - Aug 2025)",
      "bullet_points": [
        "Responsibility or achievement bullet point 1",
        "Responsibility or achievement bullet point 2"
      ]
    }}
  ],
  "education": [
    {{
      "institution": "School, College, or University Name",
      "degree": "Degree, Major, or Course",
      "location": "Location",
      "duration": "Dates/Years",
      "grade": "CGPA/GPA or Percentage Score (if any)",
      "bullet_points": [
        "Academic honors, activities, or details (if any)"
      ]
    }}
  ],
  "achievements": [
    "Achievement or honor 1",
    "Achievement or honor 2"
  ],
  "certs_projects": [
    {{
      "name": "Project Name or Certification Title",
      "role": "Role or Technologies used (e.g., React, Node.js)",
      "url": "Project link or Github link",
      "duration": "Dates/Duration",
      "bullet_points": [
        "Project description bullet point 1",
        "Project description bullet point 2"
      ]
    }}
  ]
}}

Do not include any markdown format backticks (like ```json or ```). Respond with raw JSON text only.

Raw Resume Text:
{resume_text}
"""
    system_prompt = "You are a professional resume parser. You output raw JSON only."
    
    try:
        response_text = call_groq_api(prompt, system_prompt)
        
        # Use robust brace-balancing JSON extraction
        json_str = extract_json_from_text(response_text)
        data = json.loads(json_str)
        
        # Format the fields back into clean human-readable strings
        formatted_data = {
            "name": str(data.get("name", "")).strip(),
            "summary": str(data.get("summary", "")).strip(),
            "skills_languages": _format_skills_to_text(data.get("skills_languages")),
            "skills_tools": _format_skills_to_text(data.get("skills_tools")),
            "skills_soft": _format_skills_to_text(data.get("skills_soft")),
            "experience": _format_experience_to_text(data.get("experience")),
            "education": _format_education_to_text(data.get("education")),
            "certs_projects": _format_projects_to_text(data.get("certs_projects")),
            "achievements": _format_achievements_to_text(data.get("achievements"))
        }
        return formatted_data
    except Exception as e:
        print(f"Error in LLM resume extraction: {e}")
        return {}

def load_env():
    """Load environment variables, checking os.environ, streamlit secrets, and local .env file."""
    env = {}
    
    # 1. Check system environment variables
    if "GROQ_API_KEY" in os.environ:
        env["GROQ_API_KEY"] = os.environ["GROQ_API_KEY"]
        
    # 2. Check Streamlit secrets
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            env["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
        
    # 3. Read local .env file if it exists
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    env[key.strip()] = val.strip()
    return env

def call_groq_api(prompt: str, system_prompt: str = "You are a helpful career assistant.") -> str:
    """Call the Groq API using requests. Falls back to a local model template if it fails."""
    env = load_env()
    api_key = env.get("GROQ_API_KEY", "")
    
    # If the user hasn't configured their API key, trigger fallback immediately
    if not api_key or api_key == "your_groq_api_key_here":
        return get_fallback_response(prompt)
        
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            # If rate limited or error on 70b, silently fallback to 8b-instant
            print(f"Groq API Error on 70b: {response.status_code} - {response.text}. Retrying with 8b-instant...")
            data["model"] = "llama-3.1-8b-instant"
            fallback_response = requests.post(url, headers=headers, json=data, timeout=15)
            if fallback_response.status_code == 200:
                result = fallback_response.json()
                return result["choices"][0]["message"]["content"].strip()
            print(f"Groq API Error on 8b fallback: {fallback_response.status_code} - {fallback_response.text}")
            return get_fallback_response(prompt)
    except Exception as e:
        print(f"Network error in Groq API: {e}")
        return get_fallback_response(prompt)

def get_fallback_response(prompt: str) -> str:
    """Provide rule-based/template-based responses if the Groq API is unavailable."""
    prompt_lower = prompt.lower()
    
    # Check if we are generating an outreach message
    if "outreach" in prompt_lower or "cold email" in prompt_lower:
        # Extract metadata if available in prompt
        job_title = "Software Engineer"
        company = "your company"
        name = "Candidate"
        
        # Simple extraction heuristics from prompt text
        import re
        job_match = re.search(r"Job Title:\s*([^\n]+)", prompt)
        company_match = re.search(r"Company:\s*([^\n]+)", prompt)
        name_match = re.search(r"Name:\s*([^\n]+)", prompt)
        
        if job_match: job_title = job_match.group(1).strip()
        if company_match: company = company_match.group(1).strip()
        if name_match: name = name_match.group(1).strip()
        
        return f"""Subject: Quick question re: {job_title} role at {company}

Hi Hiring Team,

I hope you're doing well. 

I'm reaching out because I noticed your team is expanding its {job_title} division. As a developer with hands-on experience in building scalable technical projects and resolving complex system constraints, I felt my skills align closely with the requirements of this role. 

In my previous projects, I've designed systems that optimized data flow efficiency and integrated modern software architectures. I'd love to learn more about the team's current challenges and see if my background might be a fit.

Would you be open to a quick 5-minute chat sometime next week?

Best regards,
{name}"""

    # Default fallback
    return "This feature requires a valid Groq API key in your .env file to generate dynamic AI content."

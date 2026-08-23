"""
analyzer.py - Core Logic for ResumeIQ
Handles text extraction, formatting checks, skill matching, and ATS scoring.
"""
import re
import fitz  # PyMuPDF
from collections import Counter

# Common tech and soft skills for basic extraction
COMMON_SKILLS = {
    # Programming Languages & Core tech
    "python", "java", "c++", "c#", "javascript", "typescript", "php", "go", "rust", "swift", "kotlin", "ruby", "sql", "html", "css", "solidity", "scala",
    # Frameworks & Libraries
    "react", "angular", "vue", "node.js", "express", "django", "flask", "fastapi", "spring boot", "rails", "laravel", "hibernate", "entity framework",
    "numpy", "pandas", "scikit-learn", "tensorflow", "pytorch", "bootstrap", "tailwind", "jquery",
    # Databases & Big Data
    "postgresql", "mysql", "mongodb", "redis", "oracle", "sql server", "nosql", "hadoop", "spark", "hive", "mapreduce", "hdfs", "kafka", "pig",
    # DevOps, Cloud & Tools
    "docker", "kubernetes", "aws", "azure", "gcp", "git", "github", "gitlab", "jenkins", "ci/cd", "terraform", "ansible", "bash", "linux", "wireshark",
    "jira", "confluence", "postman", "prometheus", "grafana", "figma", "tableau", "power bi", "excel",
    # Domain concepts (QA, PMO, Business, Legal, HR, etc.)
    "agile", "scrum", "manual testing", "test automation", "selenium", "cucumber", "testng", "junit", "api testing", "regression testing", "load testing",
    "cybersecurity", "firewalls", "vpn", "penetration testing", "siem", "sap abap", "sap hana", "fiori", "autocad", "solidworks", "matlab", "ansys",
    "structural analysis", "gis", "surveying", "legal research", "litigation", "contracts", "compliance", "intellectual property",
    "copywriting", "content writing", "seo", "blogging", "graphic design", "adobe photoshop", "creative direction", "photography",
    "recruitment", "onboarding", "employee relations", "performance management", "human resources", "talent acquisition",
    "b2b", "crm", "salesforce", "negotiation", "lead generation", "account management", "operations", "supply chain", "logistics", "six sigma",
    "pmp", "budgeting", "risk management", "resource planning", "governance", "business analysis", "requirements gathering", "uml", "user stories",
    "process mapping", "smart contracts", "cryptography", "dapps", "hyperledger", "nutrition", "wellness", "fitness coaching", "physiotherapy",
    "communication", "leadership", "problem solving", "teamwork"
}

# Advanced Market Skills Knowledge Base covering all 25 classes in the dataset
MARKET_SKILLS = {
    "Data Science": ["python", "sql", "machine learning", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "aws", "docker"],
    "Java Developer": ["java", "spring boot", "sql", "hibernate", "microservices", "docker", "kubernetes", "aws", "git", "ci/cd"],
    "Software Engineer": ["python", "java", "javascript", "c++", "sql", "git", "aws", "docker", "agile", "linux"],
    "Web Designing": ["html", "css", "javascript", "react", "figma", "ui/ux", "bootstrap", "tailwind", "adobe xd"],
    "HR": ["recruitment", "onboarding", "employee relations", "performance management", "communication", "human resources", "talent acquisition"],
    "Mechanical Engineer": ["autocad", "solidworks", "matlab", "ansys", "manufacturing", "project management", "thermodynamics"],
    "Sales": ["b2b", "crm", "salesforce", "communication", "negotiation", "lead generation", "account management"],
    "Operations Manager": ["operations", "supply chain", "logistics", "agile", "six sigma", "project management", "leadership"],
    "Advocate": ["legal research", "litigation", "contracts", "corporate law", "compliance", "drafting", "advocacy", "intellectual property"],
    "Arts": ["graphic design", "illustration", "adobe photoshop", "creative direction", "fine arts", "photography", "digital art", "typography"],
    "Database": ["sql", "oracle", "database administration", "postgresql", "mysql", "performance tuning", "backup & recovery", "nosql", "mongodb", "data warehousing"],
    "Health and fitness": ["nutrition", "personal training", "wellness", "fitness coaching", "anatomy", "cardio", "health education", "physiotherapy"],
    "PMO": ["project management", "pmp", "agile", "scrum", "budgeting", "risk management", "resource planning", "jira", "governance"],
    "Business Analyst": ["business analysis", "requirements gathering", "sql", "jira", "uml", "agile", "user stories", "tableau", "process mapping", "data analysis"],
    "DotNet Developer": ["c#", ".net core", "asp.net", "entity framework", "sql server", "mvc", "web api", "javascript", "git", "azure"],
    "Automation Testing": ["selenium", "java", "python", "test automation", "cucumber", "testng", "junit", "jenkins", "git", "api testing", "postman"],
    "Network Security Engineer": ["firewalls", "cisco", "network security", "routing & switching", "vpn", "cybersecurity", "wireshark", "penetration testing", "siem"],
    "SAP Developer": ["sap abap", "sap hana", "sap erp", "fiori", "abap oo", "sapui5", "sap gateway", "sap workflows", "sql"],
    "Civil Engineer": ["autocad", "civil engineering", "structural analysis", "concrete design", "project estimation", "gis", "construction management", "surveying"],
    "Python Developer": ["python", "django", "flask", "fastapi", "sql", "git", "docker", "rest apis", "postgresql", "celery", "numpy", "pandas"],
    "Copywriter": ["copywriting", "content writing", "seo", "blogging", "editing", "proofreading", "creative writing", "social media marketing", "brand messaging"],
    "DevOps Engineer": ["docker", "kubernetes", "jenkins", "ansible", "terraform", "aws", "git", "linux", "ci/cd", "bash", "prometheus", "grafana"],
    "Hadoop": ["hadoop", "spark", "hive", "mapreduce", "hdfs", "scala", "python", "sql", "big data", "pig", "kafka"],
    "ETL Developer": ["etl", "informatica", "sql", "data warehousing", "talend", "data integration", "ssis", "oracle", "data modeling", "pl/sql"],
    "Blockchain": ["solidity", "ethereum", "blockchain", "smart contracts", "cryptography", "web3.js", "rust", "go", "dapps", "hyperledger"],
    "Testing": ["manual testing", "test cases", "regression testing", "bug tracking", "jira", "software quality assurance", "system testing"]
}

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts plain text from a PDF file using PyMuPDF.
    Includes basic error handling for corrupted or invalid PDFs.
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return "ERROR_INVALID_PDF: The uploaded file could not be parsed."
    return text

def extract_contact_info(text: str) -> dict:
    """Uses regex to find standard contact information."""
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    linkedin_pattern = r'linkedin\.com/in/[a-zA-Z0-9-]+'
    
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    linkedins = re.findall(linkedin_pattern, text.lower())
    
    return {
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None,
        "linkedin": linkedins[0] if linkedins else None
    }

def extract_skills(text: str, custom_skill_list: list = None) -> list:
    """Finds matching skills from the text based on a predefined list."""
    text_lower = text.lower()
    found_skills = set()
    
    skills_to_check = COMMON_SKILLS
    if custom_skill_list:
        skills_to_check = set([s.lower() for s in custom_skill_list])
        
    for skill in skills_to_check:
        # Use word boundaries for exact matches
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill.title())
            
    return list(found_skills)

def check_formatting(text: str, pdf_path: str = None) -> list:
    """
    Checks for common formatting issues in the resume.
    Returns a list of dicts: [{'issue': 'description', 'severity': 'high/medium/low'}]
    """
    issues = []
    text_lower = text.lower()
    
    # 1. Contact Info check
    contacts = extract_contact_info(text)
    if not contacts["email"] and not contacts["phone"]:
        issues.append({"issue": "Missing contact information (Email/Phone).", "severity": "high"})
    if not contacts["linkedin"]:
        issues.append({"issue": "No LinkedIn profile found. Recruiters expect this.", "severity": "medium"})
        
    # 2. Length & Advanced Font Size Check (PyMuPDF dict extraction)
    if pdf_path:
        try:
            doc = fitz.open(pdf_path)
            if doc.page_count > 2:
                issues.append({"issue": f"Resume is {doc.page_count} pages. Keep it to 1-2 pages maximum.", "severity": "high"})
                
            # Extract font sizes
            font_sizes = []
            for page in doc:
                blocks = page.get_text("dict").get("blocks", [])
                for b in blocks:
                    if b.get("type") == 0:  # text block
                        for l in b.get("lines", []):
                            for s in l.get("spans", []):
                                size = s.get("size")
                                text_str = s.get("text", "").strip()
                                # Only count meaningful text spans, ignore empty spaces
                                if size and len(text_str) > 2:
                                    font_sizes.append(size)
            
            if font_sizes:
                # Find the most common font size (mode) which is likely the body text
                body_font_size = Counter(font_sizes).most_common(1)[0][0]
                if body_font_size < 10:
                    issues.append({"issue": f"Body font size is approx {body_font_size:.1f}pt. This is too small for readability. Increase to 10pt-12pt.", "severity": "high"})
                elif body_font_size > 12.5:
                    issues.append({"issue": f"Body font size is approx {body_font_size:.1f}pt. This is unusually large and wastes space. Decrease to 10pt-12pt.", "severity": "medium"})
        except Exception as e:
            print(f"Font extraction error: {e}")
        
    # 3. Missing Sections
    if "experience" not in text_lower and "employment" not in text_lower and "work history" not in text_lower:
        issues.append({"issue": "No 'Experience' section detected.", "severity": "high"})
    if "education" not in text_lower:
        issues.append({"issue": "No 'Education' section detected.", "severity": "high"})
        
    # 4. Action verbs / Metrics Check
    action_verbs = ["achieved", "improved", "developed", "managed", "created", "led", "increased", "decreased", "resolved"]
    found_verbs = [v for v in action_verbs if v in text_lower]
    if len(found_verbs) < 3:
        issues.append({"issue": "Low usage of strong action verbs (e.g., achieved, developed, led).", "severity": "medium"})
        
    numbers = re.findall(r'\b\d+%\b|\$\d+', text)
    if len(numbers) < 2:
        issues.append({"issue": "Lack of quantifiable metrics (%, $). Try to quantify your achievements.", "severity": "high"})
        
    return issues

def compute_ats_score(resume_text: str, jd_text: str) -> dict:
    """
    Computes a basic rule-based ATS match score against a Job Description.
    """
    if not jd_text.strip():
        return {"score": 0, "details": "No Job Description provided."}
        
    # 1. Skill Match
    # Extract words from JD that look like skills (simple heuristic: common tech words)
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    
    if not jd_skills:
        return {"score": 50, "details": "Job Description is too short or doesn't list standard technical skills."}
        
    matched_skills = set([s.lower() for s in resume_skills]) & set([s.lower() for s in jd_skills])
    missing_skills = set([s.lower() for s in jd_skills]) - set([s.lower() for s in resume_skills])
    
    skill_score = (len(matched_skills) / len(jd_skills)) * 100
    
    # 2. Keyword Density Match
    # Check general important keywords (longer than 4 chars) from JD in Resume
    jd_words = re.findall(r'\b[a-zA-Z]{5,}\b', jd_text.lower())
    # Get top 20 most common words in JD to use as keywords
    common_jd_words = [word for word, count in Counter(jd_words).most_common(20) if word not in ["their", "there", "about", "which", "would", "other"]]
    
    resume_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', resume_text.lower()))
    keyword_matches = [w for w in common_jd_words if w in resume_words]
    
    density_score = (len(keyword_matches) / len(common_jd_words)) * 100 if common_jd_words else 0
    
    # Final Score (Weighted average)
    final_score = int((skill_score * 0.7) + (density_score * 0.3))
    
    return {
        "score": min(final_score, 100),
        "matched_skills": list(matched_skills),
        "missing_skills": list(missing_skills),
        "keyword_matches": keyword_matches
    }

def compute_general_score(text: str, issues: list, skills: list) -> dict:
    """
    Computes a general 'Resume Health' score out of 100 based on formatting,
    completeness, and impact (metrics, action verbs).
    """
    score = 100
    breakdown = []
    
    # 1. Formatting Deductions
    for issue in issues:
        if issue["severity"] == "high":
            score -= 15
            breakdown.append({"item": issue["issue"], "impact": -15})
        elif issue["severity"] == "medium":
            score -= 5
            breakdown.append({"item": issue["issue"], "impact": -5})
            
    # 2. Skill Bonuses
    if len(skills) > 15:
        score += 5
        breakdown.append({"item": "Excellent variety of skills listed", "impact": "+5"})
    elif len(skills) == 0:
        score -= 20
        breakdown.append({"item": "No recognizable technical skills found", "impact": -20})
        
    # 3. Action Verbs & Metrics (Checking raw text for positive signals)
    text_lower = text.lower()
    action_verbs = ["achieved", "improved", "developed", "managed", "created", "led", "increased", "decreased", "resolved", "spearheaded", "architected", "optimized"]
    found_verbs = [v for v in action_verbs if v in text_lower]
    
    if len(found_verbs) >= 5:
        score += 10
        breakdown.append({"item": "Strong usage of impactful action verbs", "impact": "+10"})
        
    numbers = re.findall(r'\b\d+%\b|\$\d+', text)
    if len(numbers) >= 3:
        score += 10
        breakdown.append({"item": "Excellent use of quantified metrics (%, $)", "impact": "+10"})
        
    # Ensure score stays between 0 and 100
    final_score = max(0, min(100, score))
    
    return {
        "score": final_score,
        "breakdown": breakdown
    }

def get_market_skill_gaps(predicted_role: str, current_skills: list) -> dict:
    """
    Compares the user's skills against the MARKET_SKILLS dictionary to find missing high-value skills.
    """
    # Try exact match, otherwise try to find a substring match, otherwise fallback to generic tech
    role_skills = MARKET_SKILLS.get(predicted_role, [])
    if not role_skills:
        for k, v in MARKET_SKILLS.items():
            if k.lower() in predicted_role.lower() or predicted_role.lower() in k.lower():
                role_skills = v
                break
                
    if not role_skills:
        # Generic Software/Tech fallback
        role_skills = ["python", "sql", "git", "aws", "docker", "agile", "communication", "problem solving", "ci/cd", "javascript", "react", "cloud"]
        
    current_lower = [s.lower() for s in current_skills]
    matched = [s.title() for s in role_skills if s.lower() in current_lower]
    missing = [s.title() for s in role_skills if s.lower() not in current_lower]
    
    return {
        "matched": matched,
        "missing": missing
    }

def extract_bullet_points(text: str) -> list:
    """
    Extracts individual bullet points / line items from the resume.
    Looks for lines starting with bullets, dashes, or numbered items.
    """
    lines = text.split("\n")
    bullets = []
    for line in lines:
        cleaned = line.strip()
        # Match lines starting with bullet chars, dashes, numbers, or that look like accomplishments
        if cleaned and len(cleaned) > 15:  # skip very short lines (headers)
            if re.match(r'^[\u2022\u25CF\u25CB\u25AA\u25AB\-\*\>\|]', cleaned):
                bullets.append(re.sub(r'^[\u2022\u25CF\u25CB\u25AA\u25AB\-\*\>\|]+\s*', '', cleaned))
            elif re.match(r'^\d+[\.\)]', cleaned):
                bullets.append(re.sub(r'^\d+[\.\)]\s*', '', cleaned))
            elif any(v in cleaned.lower()[:20] for v in ["developed", "managed", "created", "designed", "built", "implemented", "led", "achieved", "increased", "reduced"]):
                bullets.append(cleaned)
    return bullets if bullets else [line.strip() for line in lines if len(line.strip()) > 30][:10]

def compute_section_scores(text: str) -> dict:
    """
    Evaluates each resume section individually and returns a score (0-100) per section.
    """
    text_lower = text.lower()
    scores = {}
    
    # Contact Info
    contacts = extract_contact_info(text)
    contact_score = 0
    if contacts["email"]: contact_score += 40
    if contacts["phone"]: contact_score += 30
    if contacts["linkedin"]: contact_score += 30
    scores["Contact Info"] = contact_score
    
    # Skills
    skills = extract_skills(text)
    scores["Skills"] = min(100, len(skills) * 8)
    
    # Experience
    exp_score = 0
    if "experience" in text_lower or "employment" in text_lower:
        exp_score += 40
    action_verbs = ["achieved", "improved", "developed", "managed", "created", "led", "increased", "decreased", "resolved", "spearheaded", "architected", "optimized"]
    found_verbs = [v for v in action_verbs if v in text_lower]
    exp_score += min(40, len(found_verbs) * 8)
    numbers = re.findall(r'\b\d+%\b|\$\d+', text)
    exp_score += min(20, len(numbers) * 5)
    scores["Experience"] = min(100, exp_score)
    
    # Education
    edu_score = 0
    if "education" in text_lower: edu_score += 50
    edu_keywords = ["bachelor", "master", "b.tech", "m.tech", "b.e", "m.e", "ph.d", "diploma", "university", "college", "degree", "gpa", "cgpa"]
    found_edu = [k for k in edu_keywords if k in text_lower]
    edu_score += min(50, len(found_edu) * 10)
    scores["Education"] = min(100, edu_score)
    
    # Projects
    proj_score = 0
    if "project" in text_lower: proj_score += 40
    proj_keywords = ["built", "developed", "designed", "github", "deployed", "implemented"]
    found_proj = [k for k in proj_keywords if k in text_lower]
    proj_score += min(60, len(found_proj) * 12)
    scores["Projects"] = min(100, proj_score)
    
    return scores

def categorize_skills(skills: list) -> dict:
    """
    Groups extracted skills into categories for a radar chart.
    """
    categories = {
        "Languages": {"python", "java", "c++", "c#", "javascript", "typescript", "go", "rust", "swift", "kotlin", "ruby", "php"},
        "Frameworks": {"react", "angular", "vue", "node.js", "express", "django", "flask", "fastapi", "spring boot", "rails", "laravel"},
        "Cloud & DevOps": {"aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "ci/cd", "terraform", "linux", "bash"},
        "Data & ML": {"machine learning", "data analysis", "artificial intelligence", "nlp", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "tableau", "power bi", "sql", "mysql", "postgresql", "mongodb", "redis"},
        "Soft Skills": {"communication", "leadership", "problem solving", "teamwork", "agile", "scrum"}
    }
    
    result = {}
    for cat, cat_skills in categories.items():
        matched = [s for s in skills if s.lower() in cat_skills]
        max_possible = len(cat_skills)
        result[cat] = {
            "matched": matched,
            "count": len(matched),
            "score": min(100, int((len(matched) / max(1, min(5, max_possible))) * 100))
        }
    return result

def extract_resume_features(text: str, pdf_path: str = None) -> dict:
    """
    Extracts numerical features from the resume for the ATS Health Score Engine.
    Returns a dict of features used by compute_health_score() in ml_model.py.
    """
    from ml_model import compute_tfidf_skill_score

    text_lower = text.lower()
    skills = extract_skills(text)
    issues = check_formatting(text, pdf_path)

    action_verbs = ["achieved", "improved", "developed", "managed", "created", "led",
                    "increased", "decreased", "resolved", "spearheaded", "architected", "optimized"]
    found_verbs = [v for v in action_verbs if v in text_lower]

    numbers = re.findall(r'\b\d+%\b|\$\d+', text)

    formatting_penalty = sum(15 if i["severity"] == "high" else 5 for i in issues)

    sections = ["experience", "education", "skills", "project", "summary", "objective"]
    found_sections  = sum(1 for s in sections if s in text_lower)
    section_completeness = (found_sections / len(sections)) * 100

    # Keyword density: ratio of unique meaningful words to total words
    all_words    = re.findall(r'\b[a-zA-Z]{4,}\b', text_lower)
    unique_words = set(all_words)
    keyword_density = min(100, (len(unique_words) / max(1, len(all_words))) * 100)

    # IR-based skill score: TF-IDF cosine similarity vs live job corpus (ml_model.py)
    tfidf_skill_score = compute_tfidf_skill_score(text)

    return {
        "skill_count":        len(skills),         # kept for legacy display
        "tfidf_skill_score":  tfidf_skill_score,   # used by compute_health_score()
        "keyword_density":    keyword_density,
        "action_verb_count":  len(found_verbs),
        "metrics_count":      len(numbers),
        "formatting_penalty": min(50, formatting_penalty),
        "section_completeness": section_completeness
    }

def calculate_yoe(text: str) -> float:
    """
    State-of-the-Art Estimator for Years of Experience (YoE)
    Extracts date ranges from resume text and calculates total duration.
    """
    import datetime
    
    # Common date formats: 
    # "Jan 2019 - Dec 2021", "2015 to 2018", "05/2020 - Present"
    date_range_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?[a-z]*\s*\d{4}|\d{1,2}/\d{4})\s*(?:-|to|–)\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?[a-z]*\s*\d{4}|\d{1,2}/\d{4}|Present|Current|Now)'
    
    matches = re.findall(date_range_pattern, text, re.IGNORECASE)
    
    total_months = 0
    current_year = datetime.datetime.now().year
    
    for start_str, end_str in matches:
        try:
            # Extract year from start
            start_year_match = re.search(r'\d{4}', start_str)
            if not start_year_match: continue
            start_year = int(start_year_match.group())
            
            # Extract year from end
            if end_str.lower() in ['present', 'current', 'now']:
                end_year = current_year
            else:
                end_year_match = re.search(r'\d{4}', end_str)
                if not end_year_match: continue
                end_year = int(end_year_match.group())
                
            # Basic sanity check
            if 1950 <= start_year <= current_year and start_year <= end_year <= current_year + 5:
                # Add 1 to be inclusive (e.g. 2020-2020 is 1 year)
                total_months += (end_year - start_year) * 12 + 12
        except Exception:
            continue
            
    # Convert months to years
    total_years = round(total_months / 12, 1)
    # Cap at realistic number to avoid bad parse blowups
    return min(total_years, 35.0)

"""
Smart Interview and PDF Generator for ResumeIQ.

Analyzes ML pipeline output to generate targeted interview questions,
then builds a clean, professional PDF resume from the user's answers
and original resume text.
"""

from fpdf import FPDF
import re
import os
import tempfile


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _extract_name(text: str) -> str:
    """Return the first non-empty line of *text* as the candidate's name."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "Your Name"


def _extract_section(text: str, section_name: str) -> str:
    """Try to extract the content between *section_name* and the next section header.

    Section headers are identified as lines that look like all-caps labels,
    title-case labels, or lines ending with a colon.
    """
    # Build a pattern that matches the section header (case-insensitive)
    header_pattern = re.compile(
        rf"(?i)^\s*{re.escape(section_name)}\s*:?\s*$", re.MULTILINE
    )
    match = header_pattern.search(text)
    if not match:
        return ""

    start = match.end()

    # Look for the next section-like header after the matched one
    next_header = re.compile(
        r"^\s*[A-Z][A-Za-z &/]+\s*:?\s*$", re.MULTILINE
    )
    next_match = next_header.search(text, start)
    end = next_match.start() if next_match else len(text)

    return text[start:end].strip()


def _safe_text(text: str) -> str:
    """Replace non-latin characters so FPDF can render the string."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ---------------------------------------------------------------------------
# 1. Interview-question generator
# ---------------------------------------------------------------------------

def generate_interview_questions(
    issues: list[str],
    skills: list[str],
    market_gaps: dict,
    bullet_results: list[dict],
) -> list[str]:
    """Analyze ML output and dynamically generate 5-7 targeted questions.

    Parameters
    ----------
    issues : list[str]
        List of issue strings detected by the resume analyzer
        (e.g. ``["No email found", "No Experience section"]``).
    skills : list[str]
        Skills already present on the resume.
    market_gaps : dict
        Dictionary with at least a ``"missing_skills"`` key whose value is a
        list of in-demand skills the candidate lacks.
    bullet_results : list[dict]
        Per-bullet analysis dicts, each containing at least a ``"label"`` key
        (``"Strong"`` or ``"Weak"``).

    Returns
    -------
    list[str]
        Between 5 and 7 question strings.
    """
    questions: list[str] = []

    # Extract the 'issue' text from each issue dictionary
    issues_lower = [i.get("issue", "").lower() for i in issues]
    issues_text = " ".join(issues_lower)

    contact_missing = any(
        kw in issues_text
        for kw in ("no email", "no phone", "contact", "missing email", "missing phone")
    )
    if contact_missing:
        questions.append(
            "What is your email address and phone number?"
        )

    linkedin_missing = any("linkedin" in issue for issue in issues_lower)
    if linkedin_missing:
        questions.append(
            "What is your LinkedIn profile URL?"
        )

    # --- Experience section ----------------------------------------------------
    if any("no experience" in issue for issue in issues_lower):
        questions.append(
            "Please describe your most recent work experience or internship, "
            "including your role, company name, and key responsibilities."
        )

    # --- Metrics / measurable results ------------------------------------------
    metrics_lacking = any(
        kw in issue for issue in issues_lower
        for kw in ("metric", "quantif", "measurable", "no numbers")
    )
    if metrics_lacking:
        questions.append(
            "Can you describe a project where you achieved a measurable result? "
            "(e.g., Increased sales by 20%, Reduced load time by 40%)"
        )

    # --- Action verbs ----------------------------------------------------------
    weak_verbs = any(
        kw in issue for issue in issues_lower
        for kw in ("action verb", "weak verb", "passive")
    )
    if weak_verbs:
        questions.append(
            "Describe your biggest professional achievement using strong action "
            "words (e.g., Led, Developed, Optimized)."
        )

    # --- Market-gap skills -----------------------------------------------------
    missing_skills = market_gaps.get("missing_skills", [])
    if missing_skills:
        missing_skills_str = ", ".join(missing_skills[:6])
        questions.append(
            f"Your profile is missing in-demand skills like {missing_skills_str}. "
            "Do you have experience with any of these? If yes, briefly describe."
        )

    # --- Weak bullet points ----------------------------------------------------
    has_weak = any(
        b.get("label", "").lower() == "weak" for b in bullet_results
    )
    if has_weak:
        questions.append(
            "Some of your bullet points lack impact. Rewrite your most important "
            "accomplishment in one powerful sentence."
        )

    # --- Pad to at least 5 questions ------------------------------------------
    generic_questions = [
        "In 2-3 sentences, write a professional summary/objective for your resume.",
        "List any certifications, awards, or notable projects not already on your resume.",
        "What are your top three technical or professional strengths?",
        "Describe a challenge you overcame at work and the outcome.",
        "What career goals would you like your resume to reflect?",
    ]

    for gq in generic_questions:
        if len(questions) >= 5:
            break
        if gq not in questions:
            questions.append(gq)

    # Cap at 7
    return questions[:7]


# ---------------------------------------------------------------------------
# 2. Enhanced PDF generator
# ---------------------------------------------------------------------------

def generate_enhanced_pdf(
    original_text: str,
    answers: dict[str, str],
    skills: list[str],
    predicted_role: str,
    output_path: str = "enhanced_resume.pdf",
    font_family: str = "Helvetica",
    base_font_size: int = 10,
    margin: int = 15,
    primary_color_name: str = "Dark Blue",
    page_size: str = "A4",
    template_style: str = "Classic Modern"
) -> str:
    """Generate a clean, professional PDF resume with customizable layout and styles.

    Parameters
    ----------
    original_text : str
        The raw text of the candidate's original resume.
    answers : dict[str, str]
        Mapping of question → answer collected during the interview step.
    skills : list[str]
        Aggregated list of skills (extracted + user-provided).
    predicted_role : str
        The role predicted by the ML classifier (used as a subtitle hint).
    output_path : str, optional
        Destination file path for the PDF. Defaults to 'enhanced_resume.pdf'.
    font_family : str, optional
        Font family name ('Helvetica', 'Times', 'Courier'). Defaults to 'Helvetica'.
    base_font_size : int, optional
        Base font size in points. Defaults to 10.
    margin : int, optional
        Page margin in millimeters. Defaults to 15.
    primary_color_name : str, optional
        Color theme name ('Dark Blue', 'Black', 'Green', 'Crimson Red', 'Slate Grey').
    page_size : str, optional
        Page size ('A4', 'Letter'). Defaults to 'A4'.
    template_style : str, optional
        Template theme ('Classic Modern', 'Minimalist', 'Executive').

    Returns
    -------
    str
        The absolute path of the generated PDF file.
    """
    COLOR_MAP = {
        "Dark Blue": (44, 62, 80),
        "Black": (0, 0, 0),
        "Green": (6, 95, 70),
        "Crimson Red": (153, 27, 27),
        "Slate Grey": (71, 85, 105),
    }
    primary_color = COLOR_MAP.get(primary_color_name, (44, 62, 80))
    BLACK = (0, 0, 0)

    pdf = FPDF(format=page_size)
    pdf.set_auto_page_break(auto=True, margin=margin)
    pdf.add_page()
    pdf.set_margins(margin, margin, margin)

    # -- Extract core details --------------------------------------------------
    name = _safe_text(_extract_name(original_text))

    # Try to pull email and phone from original text first
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", original_text)
    phone_match = re.search(r"[\+]?[\d\s\-().]{7,15}", original_text)

    email = email_match.group(0) if email_match else ""
    phone = phone_match.group(0).strip() if phone_match else ""

    # Override with interview answers if the user provided contact info
    for q, a in answers.items():
        q_lower = q.lower()
        if "email" in q_lower and "phone" in q_lower and a.strip():
            ans_email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", a)
            ans_phone = re.search(r"[\+]?[\d\s\-().]{7,15}", a)
            if ans_email:
                email = ans_email.group(0)
            if ans_phone:
                phone = ans_phone.group(0).strip()

    contact_parts = [p for p in (email, phone) if p]
    contact_line = _safe_text(" | ".join(contact_parts)) if contact_parts else ""

    # -- Header rendering based on style ---------------------------------------
    if template_style == "Classic Modern":
        # Centered Header
        pdf.set_font(font_family, "B", base_font_size * 2 + 4)
        pdf.set_text_color(*BLACK)
        pdf.cell(0, 10, name.upper(), new_x="LMARGIN", new_y="NEXT", align="C")

        if contact_line:
            pdf.set_font(font_family, "", base_font_size)
            pdf.cell(0, 6, contact_line, new_x="LMARGIN", new_y="NEXT", align="C")

        if predicted_role:
            pdf.set_font(font_family, "I", base_font_size + 1)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 6, _safe_text(predicted_role), new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.set_text_color(*BLACK)

        pdf.ln(2)
        pdf.set_draw_color(*primary_color)
        pdf.set_line_width(0.5)
        pdf.line(margin, pdf.get_y(), pdf.w - margin, pdf.get_y())
        pdf.ln(4)

    elif template_style == "Minimalist":
        # Left-aligned Header, clean design, no header lines
        pdf.set_font(font_family, "B", base_font_size * 2 + 2)
        pdf.set_text_color(*BLACK)
        pdf.cell(0, 10, name, new_x="LMARGIN", new_y="NEXT", align="L")

        if contact_line:
            pdf.set_font(font_family, "", base_font_size - 0.5)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 5, contact_line, new_x="LMARGIN", new_y="NEXT", align="L")

        if predicted_role:
            pdf.set_font(font_family, "", base_font_size)
            pdf.set_text_color(*primary_color)
            pdf.cell(0, 5, _safe_text(predicted_role), new_x="LMARGIN", new_y="NEXT", align="L")
            pdf.set_text_color(*BLACK)
            
        pdf.ln(6)

    else:  # Executive
        # Split header: Name on left, Contact on right
        pdf.set_font(font_family, "B", base_font_size * 2 + 2)
        pdf.set_text_color(*primary_color)
        
        # Calculate width for name
        current_y = pdf.get_y()
        pdf.cell(0, 10, name.upper(), align="L")
        
        # Contact line right-aligned on same line if possible
        if contact_line:
            pdf.set_y(current_y + 2)
            pdf.set_font(font_family, "", base_font_size - 0.5)
            pdf.set_text_color(*BLACK)
            pdf.cell(0, 6, contact_line, align="R", new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.ln(10)
            
        if predicted_role:
            pdf.set_font(font_family, "I", base_font_size + 1)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 6, _safe_text(predicted_role), new_x="LMARGIN", new_y="NEXT", align="L")
            
        pdf.ln(2)
        pdf.set_draw_color(*primary_color)
        pdf.set_line_width(1.0) # Thick bar
        pdf.line(margin, pdf.get_y(), pdf.w - margin, pdf.get_y())
        pdf.ln(4)

    # -- Helper: add a section -------------------------------------------------
    def _add_section(title: str, body: str) -> None:
        """Render a section header + body text block with proper ATS formatting."""
        if not body.strip():
            return

        if template_style == "Minimalist":
            pdf.set_font(font_family, "B", base_font_size + 1.5)
            pdf.set_text_color(*BLACK)
            pdf.cell(0, 7, _safe_text(title.title()), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.5)
        elif template_style == "Executive":
            pdf.set_font(font_family, "B", base_font_size + 2)
            pdf.set_text_color(*primary_color)
            pdf.cell(0, 7, _safe_text(title.upper()), new_x="LMARGIN", new_y="NEXT")
            pdf.set_draw_color(*primary_color)
            pdf.set_line_width(0.6)
            pdf.line(margin, pdf.get_y(), pdf.w - margin, pdf.get_y())
            pdf.ln(2.5)
        else: # Classic Modern
            pdf.set_font(font_family, "B", base_font_size + 2)
            pdf.set_text_color(*primary_color)
            pdf.cell(0, 7, _safe_text(title.upper()), new_x="LMARGIN", new_y="NEXT")
            pdf.set_draw_color(180, 180, 180)
            pdf.set_line_width(0.2)
            pdf.line(margin, pdf.get_y(), pdf.w - margin, pdf.get_y())
            pdf.ln(2)

        pdf.set_font(font_family, "", base_font_size)
        pdf.set_text_color(*BLACK)
        
        # Parse text into bullet points and paragraphs
        for line in body.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                clean_line = _safe_text(line.lstrip("-•* ").strip())
                pdf.set_x(margin + 5)
                pdf.multi_cell(0, 5, "\x95  " + clean_line)
            else:
                pdf.set_x(margin)
                pdf.multi_cell(0, 5.5, _safe_text(line))
                
        pdf.ln(4)

    # -- Professional Summary --------------------------------------------------
    summary_answer = ""
    for q, a in answers.items():
        if "summary" in q.lower() or "objective" in q.lower():
            summary_answer = a.strip()
            break

    if not summary_answer:
        paragraphs = [p.strip() for p in original_text.split("\n\n") if p.strip()]
        summary_answer = paragraphs[0] if paragraphs else ""

    _add_section("Professional Summary", summary_answer)

    # -- Skills ----------------------------------------------------------------
    skills_text = ", ".join(skills) if skills else ""
    _add_section("Skills", skills_text)

    # -- Experience ------------------------------------------------------------
    experience_body = _extract_section(original_text, "Experience")
    for q, a in answers.items():
        if "experience" in q.lower() or "internship" in q.lower():
            if a.strip():
                experience_body = (
                    f"{experience_body}\n\n{a.strip()}" if experience_body else a.strip()
                )
            break
    _add_section("Experience", experience_body)

    # -- Education -------------------------------------------------------------
    education_body = _extract_section(original_text, "Education")
    _add_section("Education", education_body)

    # -- Achievements ----------------------------------------------------------
    achievements_parts: list[str] = []
    for q, a in answers.items():
        q_lower = q.lower()
        if any(kw in q_lower for kw in ("measurable", "achievement", "accomplishment")):
            if a.strip():
                achievements_parts.append(a.strip())
    _add_section("Achievements", "\n".join(achievements_parts))

    # -- Certifications & Projects ---------------------------------------------
    certs_answer = ""
    for q, a in answers.items():
        if "certification" in q.lower() or "project" in q.lower() or "award" in q.lower():
            if a.strip():
                certs_answer = a.strip()
            break

    certs_body = _extract_section(original_text, "Certifications") or ""
    projects_body = _extract_section(original_text, "Projects") or ""
    combined = "\n".join(filter(None, [certs_body, projects_body, certs_answer]))
    _add_section("Certifications & Projects", combined)

    # -- Write PDF -------------------------------------------------------------
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    pdf.output(output_path)
    return os.path.abspath(output_path)

# ---------------------------------------------------------------------------
# 3. Automated Cover Letter Generator
# ---------------------------------------------------------------------------

def generate_cover_letter_pdf(
    original_text: str,
    skills: list[str],
    job_title: str,
    company: str,
    output_path: str = "cover_letter.pdf"
) -> str:
    """Generate a tailored Cover Letter PDF for a specific job match."""
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(20, 20, 20)
    
    BLACK = (0, 0, 0)
    name = _safe_text(_extract_name(original_text))
    
    # Try to pull email and phone
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", original_text)
    phone_match = re.search(r"[\+]?[\d\s\-().]{7,15}", original_text)
    email = email_match.group(0) if email_match else "your.email@example.com"
    phone = phone_match.group(0).strip() if phone_match else "(555) 555-5555"

    # Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*BLACK)
    pdf.cell(0, 10, name, new_x="LMARGIN", new_y="NEXT", align="L")
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, _safe_text(email), new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.cell(0, 5, _safe_text(phone), new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(10)
    
    import datetime
    today = datetime.datetime.now().strftime("%B %d, %Y")
    pdf.cell(0, 5, today, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 5, _safe_text(f"Hiring Manager"), new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.cell(0, 5, _safe_text(company), new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(10)
    
    # Body
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 5, "Dear Hiring Manager,", new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(5)
    
    p1 = f"I am writing to express my strong interest in the {job_title} position at {company}. With my background in technology and a proven track record of delivering results, I am confident in my ability to make an immediate impact on your team."
    pdf.multi_cell(0, 6, _safe_text(p1))
    pdf.ln(5)
    
    top_skills = ", ".join(skills[:5]) if skills else "software development, problem-solving, and continuous learning"
    p2 = f"Throughout my career, I have developed expertise in {top_skills}. My experience aligns closely with the responsibilities of the {job_title} role, particularly my ability to adapt to new technologies and drive successful project outcomes."
    pdf.multi_cell(0, 6, _safe_text(p2))
    pdf.ln(5)
    
    p3 = f"What draws me to {company} is your commitment to innovation and excellence. I am excited about the opportunity to bring my technical skills and collaborative mindset to your engineering organization, and to contribute to the high-quality products your team builds."
    pdf.multi_cell(0, 6, _safe_text(p3))
    pdf.ln(5)
    
    p4 = "I would welcome the opportunity to discuss how my experience and vision align with the goals of your team. Thank you for considering my application. I have attached my resume for your review and look forward to the possibility of speaking with you soon."
    pdf.multi_cell(0, 6, _safe_text(p4))
    pdf.ln(10)
    
    pdf.cell(0, 6, "Sincerely,", new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, name, new_x="LMARGIN", new_y="NEXT", align="L")
    
    # Write PDF
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    pdf.output(output_path)
    return os.path.abspath(output_path)


def clean_resume_text_bullets(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line in ["•", "-", "*", "➜", "->", "=>"]:
            next_idx = i + 1
            while next_idx < len(lines) and not lines[next_idx].strip():
                next_idx += 1
            if next_idx < len(lines):
                next_line = lines[next_idx].strip()
                cleaned_lines.append(f"- {next_line}")
                i = next_idx + 1
            else:
                cleaned_lines.append(line)
                i += 1
        else:
            cleaned_lines.append(lines[i])
            i += 1
    return "\n".join(cleaned_lines)

def split_text_into_entry_blocks(text: str) -> list[str]:
    # Merge double newlines followed by bullets
    text = re.sub(r"\n\n+\s*([\-•\*])", r"\n\1", text)
    
    # Split into lines
    lines = [l.strip() for l in text.split("\n")]
    
    blocks = []
    current_block_lines = []
    has_bullets = False
    
    for line in lines:
        if not line:
            if current_block_lines:
                blocks.append("\n".join(current_block_lines))
                current_block_lines = []
                has_bullets = False
            continue
            
        is_bullet = line.startswith("-") or line.startswith("•") or line.startswith("*")
        
        if current_block_lines and not is_bullet and has_bullets:
            blocks.append("\n".join(current_block_lines))
            current_block_lines = [line]
            has_bullets = False
        else:
            current_block_lines.append(line)
            if is_bullet:
                has_bullets = True
                
    if current_block_lines:
        blocks.append("\n".join(current_block_lines))
        
    return [b.strip() for b in blocks if b.strip()]

def _parse_resume_section(text: str) -> list[dict]:
    if not text or not text.strip():
        return []
        
    # Aggressively split inline bullets from PyMuPDF raw dumps
    text = re.sub(r"([a-zA-Z0-9.,])\s*[•\-]\s*([A-Z])", r"\1\n- \2", text)
    text = re.sub(r"\s*[•]\s*", r"\n- ", text)
    
    # Clean orphaned bullet characters
    text = clean_resume_text_bullets(text)
    
    # Split into entries/blocks logically
    raw_blocks = split_text_into_entry_blocks(text)

    parsed_entries = []
    for block in raw_blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue
            
        entry = {
            "title": "",
            "subtitle": "",
            "location": "",
            "dates": "",
            "bullets": []
        }
        
        non_bullets = []
        for line in lines:
            if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                bullet_content = line.lstrip("-•* ").strip()
                if bullet_content:
                    entry["bullets"].append(bullet_content)
            else:
                non_bullets.append(line)
                
        if len(non_bullets) >= 1:
            header1 = non_bullets[0]
            if "|" in header1:
                parts = [p.strip() for p in header1.split("|", 1)]
                entry["title"] = parts[0]
                entry["subtitle"] = parts[1]
            else:
                entry["title"] = header1
                
        if len(non_bullets) >= 2:
            header2 = non_bullets[1]
            if "|" in header2:
                parts = [p.strip() for p in header2.split("|", 1)]
                entry["location"] = parts[0]
                entry["dates"] = parts[1]
            else:
                # Date detection hint: if it contains digits or year, assign to dates
                if any(char.isdigit() for char in header2) or any(w in header2.lower() for w in ["present", "current", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]):
                    entry["dates"] = header2
                else:
                    entry["location"] = header2
                
        if len(non_bullets) > 2:
            for extra in non_bullets[2:]:
                entry["bullets"].insert(0, extra)
                
        parsed_entries.append(entry)
        
    return parsed_entries


def _extract_contact_info(text: str) -> dict:
    info = {
        "email": "",
        "phone": "",
        "linkedin": "",
        "github": "",
        "website": "",
        "location": ""
    }
    if not text:
        return info
        
    # Email
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    if email_match:
        info["email"] = email_match.group(0)
        
    # Phone
    phone_match = re.search(r"[\+]?[\d\s\-().]{7,15}", text)
    if phone_match:
        info["phone"] = phone_match.group(0).strip()
        
    # LinkedIn
    linkedin_match = re.search(r"(?:linkedin\.com/in/|linkedin\.com/pub/)[a-zA-Z0-9_-]+", text, re.IGNORECASE)
    if linkedin_match:
        info["linkedin"] = linkedin_match.group(0)
    else:
        linkedin_line = re.search(r"linkedin:\s*([^\s\n]+)", text, re.IGNORECASE)
        if linkedin_line:
            info["linkedin"] = "linkedin.com/in/" + linkedin_line.group(1).strip()

    # GitHub
    github_match = re.search(r"github\.com/[a-zA-Z0-9_-]+", text, re.IGNORECASE)
    if github_match:
        info["github"] = github_match.group(0)
    else:
        github_line = re.search(r"github:\s*([^\s\n]+)", text, re.IGNORECASE)
        if github_line:
            info["github"] = "github.com/" + github_line.group(1).strip()

    # Website
    website_matches = re.findall(r"(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+/[a-zA-Z0-9_-]*)", text)
    for m in website_matches:
        m_lower = m.lower()
        if "linkedin" not in m_lower and "github" not in m_lower and "email" not in m_lower:
            info["website"] = m
            break

    # Location heuristic (lines 2-5 of the text)
    lines = [line.strip() for line in text.split("\n") if line.strip()][:5]
    for line in lines[1:]:
        if re.search(r"\b[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\b", line) or re.search(r"\b[A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+\b", line):
            if "@" not in line and len(line) < 50:
                info["location"] = line
                break

    return info


def _format_markdown_to_html(text: str) -> str:
    if not text:
        return ""
    # Convert markdown bold to HTML strong
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__(.*?)__", r"<strong>\1</strong>", text)
    # Convert markdown italic to HTML em
    text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)
    text = re.sub(r"_(.*?)_", r"<em>\1</em>", text)
    return text


def _generate_html_resume(data: dict) -> str:
    template_style = data.get("template_style", "Clean")
    font_family = data.get("font_family", "Helvetica")
    base_font_size = data.get("base_font_size", 10)
    primary_color_name = data.get("primary_color_name", "Dark Blue")
    
    # CSS Color mapping
    COLOR_MAP = {
        "Dark Blue": "#1d4ed8",
        "Black": "#000000",
        "Green": "#059669",
        "Crimson Red": "#dc2626",
        "Slate Grey": "#4b5563",
    }
    primary_color = COLOR_MAP.get(primary_color_name, "#1d4ed8")
    
    # Font family mapping
    font_family_css = "system-ui, -apple-system, sans-serif"
    font_import = ""
    if font_family == "Helvetica":
        font_family_css = "'Inter', sans-serif"
        font_import = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');"
    elif font_family == "Times":
        font_family_css = "'Playfair Display', Georgia, serif"
        font_import = "@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@400;500&display=swap');"
    elif font_family == "Courier":
        font_family_css = "'Fira Code', monospace"
        font_import = "@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;700&family=Inter:wght@400;500&display=swap');"

    # Base HTML template wrapper
    html_start = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        {font_import}
        
        :root {{
            --primary-color: {primary_color};
            --font-size-base: {base_font_size}pt;
        }}
        
        body {{
            font-family: {font_family_css};
            font-size: var(--font-size-base);
            color: #1f2937; /* gray-800 */
            line-height: 1.35;
            padding: 0;
            margin: 0;
        }}
        
        .section-title {{
            font-family: {font_family_css};
            font-weight: 700;
            color: #374151; /* gray-700 */
            border-bottom: 1px solid #d1d5db;
            margin-bottom: 0.5rem;
            padding-bottom: 0.125rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        
        a {{
            color: inherit;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        
        @media print {{
            body {{
                padding: 0;
                margin: 0;
            }}
            .break-inside-avoid {{
                break-inside: avoid !important;
                page-break-inside: avoid !important;
            }}
        }}
    </style>
</head>
<body class="bg-white">
"""

    html_end = """
</body>
</html>
"""

    # Helper to render entry items (Exp, Edu, Projects)
    def render_entries_html(entries, style="Clean"):
        if not entries:
            return ""
        html = '<div class="space-y-3.5">'
        for entry in entries:
            html += '<div class="entry-item">'
            
            # Header block (Header line + Subtitle line) wrapped to avoid breaks inside
            # and avoid breaking immediately after it.
            has_header = entry["title"] or entry["subtitle"] or entry["location"] or entry["dates"]
            if has_header:
                html += '  <div class="break-inside-avoid" style="break-after: avoid; page-break-after: avoid;">'
                # Header line
                html += '    <div class="flex justify-between items-baseline">'
                html += f'      <span class="font-bold text-gray-900">{entry["title"]}</span>'
                if entry["dates"]:
                    html += f'      <span class="text-[0.85em] text-gray-500 font-mono flex-shrink-0">{entry["dates"]}</span>'
                html += '    </div>'
                
                # Subtitle line
                if entry["subtitle"] or entry["location"]:
                    html += '    <div class="flex justify-between items-baseline text-[0.92em] text-gray-600 italic mt-0.5">'
                    html += f'      <span>{entry["subtitle"]}</span>'
                    if entry["location"]:
                        html += f'      <span class="text-[0.9em] font-normal not-italic text-gray-500">{entry["location"]}</span>'
                    html += '    </div>'
                html += '  </div>'
                
            # Bullets
            if entry["bullets"]:
                bullet_marker = "•"
                if style == "Vivid":
                    bullet_marker = '<span class="text-[var(--primary-color)] mr-1.5 flex-shrink-0 select-none">➜</span>'
                else:
                    bullet_marker = '<span class="text-gray-400 mr-2 flex-shrink-0 select-none">•</span>'
                    
                html += '  <ul class="list-none pl-3 mt-1 text-[0.95em] text-gray-700 space-y-1">'
                for bullet in entry["bullets"]:
                    formatted_bullet = _format_markdown_to_html(bullet)
                    html += f'    <li class="flex items-start break-inside-avoid">{bullet_marker}<span>{formatted_bullet}</span></li>'
                html += '  </ul>'
                
            html += '</div>'
        html += '</div>'
        return html

    def render_skills_badges(skills_str):
        if not skills_str:
            return ""
        items = [s.strip() for s in skills_str.replace("\n", ",").split(",") if s.strip()]
        return "".join([f'<span class="bg-gray-100 text-gray-800 text-[10px] px-2 py-0.5 rounded border border-gray-250 font-medium select-none">{item}</span>' for item in items])

    # Build Header Section
    contact = data.get("contact", {})
    contact_parts = []
    if contact.get("location"):
        contact_parts.append(f'<span><i class="fa fa-map-marker-alt text-gray-400 mr-1.5"></i>{contact["location"]}</span>')
    if contact.get("phone"):
        contact_parts.append(f'<span><i class="fa fa-phone text-gray-400 mr-1.5"></i>{contact["phone"]}</span>')
    if contact.get("email"):
        contact_parts.append(f'<span><i class="fa fa-envelope text-gray-400 mr-1.5"></i><a href="mailto:{contact["email"]}">{contact["email"]}</a></span>')
    if contact.get("linkedin"):
        clean_li = contact["linkedin"].replace("https://", "").replace("www.", "").replace("linkedin.com/in/", "")
        contact_parts.append(f'<span><i class="fab fa-linkedin text-gray-400 mr-1.5"></i><a href="https://linkedin.com/in/{clean_li}" target="_blank">{clean_li}</a></span>')
    if contact.get("github"):
        clean_gh = contact["github"].replace("https://", "").replace("www.", "").replace("github.com/", "")
        contact_parts.append(f'<span><i class="fab fa-github text-gray-400 mr-1.5"></i><a href="https://github.com/{clean_gh}" target="_blank">{clean_gh}</a></span>')
    if contact.get("website"):
        clean_web = contact["website"].replace("https://", "").replace("www.", "")
        contact_parts.append(f'<span><i class="fa fa-globe text-gray-400 mr-1.5"></i><a href="https://{clean_web}" target="_blank">{clean_web}</a></span>')

    name = data.get("name", "Your Name")
    predicted_role = data.get("predicted_role", "")

    # Build body based on selected style
    body_html = ""
    
    if template_style == "Clean":
        # Centered clean header
        body_html += f"""
        <div class="text-center mb-6">
            <h1 class="text-3xl font-light tracking-wide text-gray-900 uppercase font-serif" style="font-family: inherit;">{name}</h1>
            {f'<div class="text-sm tracking-wider text-gray-500 uppercase mt-1 font-mono">{predicted_role}</div>' if predicted_role else ''}
            <div class="flex flex-wrap justify-center items-center gap-x-3 gap-y-1 mt-2.5 text-xs text-gray-600">
                {' <span class="text-gray-300">|</span> '.join(contact_parts)}
            </div>
        </div>
        """
        
        # Sections
        if data.get("summary"):
            formatted_summary = _format_markdown_to_html(data["summary"])
            body_html += f"""
            <div class="mb-5 break-inside-avoid">
                <h2 class="section-title">Professional Summary</h2>
                <p class="text-justify text-[0.95em] text-gray-700">{formatted_summary}</p>
            </div>
            """
            
        if data.get("skills_languages") or data.get("skills_tools") or data.get("skills_soft"):
            body_html += f"""
            <div class="mb-5 break-inside-avoid">
                <h2 class="section-title">Skills</h2>
                <div class="text-sm text-gray-700 space-y-1">
                    {f'<div><span class="font-bold">Languages:</span> {data["skills_languages"]}</div>' if data.get("skills_languages") else ''}
                    {f'<div><span class="font-bold">Tools & Technologies:</span> {data["skills_tools"]}</div>' if data.get("skills_tools") else ''}
                    {f'<div><span class="font-bold">Soft Skills:</span> {data["skills_soft"]}</div>' if data.get("skills_soft") else ''}
                </div>
            </div>
            """
            
        if data.get("experience_parsed"):
            body_html += f"""
            <div class="mb-5">
                <h2 class="section-title">Work Experience</h2>
                {render_entries_html(data["experience_parsed"], "Clean")}
            </div>
            """
            
        if data.get("education_parsed"):
            body_html += f"""
            <div class="mb-5">
                <h2 class="section-title">Education</h2>
                {render_entries_html(data["education_parsed"], "Clean")}
            </div>
            """
            
        if data.get("achievements_parsed"):
            body_html += f"""
            <div class="mb-5">
                <h2 class="section-title">Achievements</h2>
                {render_entries_html(data["achievements_parsed"], "Clean")}
            </div>
            """
            
        if data.get("certs_projects_parsed"):
            body_html += f"""
            <div class="mb-5">
                <h2 class="section-title">Certifications & Projects</h2>
                {render_entries_html(data["certs_projects_parsed"], "Clean")}
            </div>
            """

    elif template_style == "Modern Two-Column":
        # Two-column layout
        body_html += f"""
        <div class="border-b border-gray-200 pb-4 mb-6 flex justify-between items-baseline">
            <div>
                <h1 class="text-3xl font-extrabold tracking-tight text-gray-900">{name}</h1>
                {f'<div class="text-md font-medium text-[var(--primary-color)] uppercase tracking-wider mt-1">{predicted_role}</div>' if predicted_role else ''}
            </div>
        </div>
        
        <div class="grid grid-cols-12 gap-6 items-start">
            <!-- Sidebar Column -->
            <div class="col-span-4 space-y-5">
                <!-- Contact info stack -->
                <div>
                    <h3 class="text-xs font-bold text-[var(--primary-color)] uppercase tracking-widest border-b border-gray-200 pb-1 mb-2">Contact</h3>
                    <div class="space-y-2 text-xs text-gray-600 flex flex-col">
                        {f'<div><i class="fa fa-map-marker-alt text-[var(--primary-color)] w-4 text-center"></i> {contact["location"]}</div>' if contact.get("location") else ''}
                        {f'<div><i class="fa fa-phone text-[var(--primary-color)] w-4 text-center"></i> {contact["phone"]}</div>' if contact.get("phone") else ''}
                        {f'<div><i class="fa fa-envelope text-[var(--primary-color)] w-4 text-center"></i> <a href="mailto:{contact["email"]}">{contact["email"]}</a></div>' if contact.get("email") else ''}
                        {f'<div><i class="fab fa-linkedin text-[var(--primary-color)] w-4 text-center"></i> <a href="https://{contact["linkedin"]}" target="_blank">LinkedIn</a></div>' if contact.get("linkedin") else ''}
                        {f'<div><i class="fab fa-github text-[var(--primary-color)] w-4 text-center"></i> <a href="https://{contact["github"]}" target="_blank">GitHub</a></div>' if contact.get("github") else ''}
                        {f'<div><i class="fa fa-globe text-[var(--primary-color)] w-4 text-center"></i> <a href="https://{contact["website"]}" target="_blank">Website</a></div>' if contact.get("website") else ''}
                    </div>
                </div>
                
                <!-- Skills stack -->
                {(data.get("skills_languages") or data.get("skills_tools") or data.get("skills_soft")) and f'''
                <div>
                    <h3 class="text-xs font-bold text-[var(--primary-color)] uppercase tracking-widest border-b border-gray-200 pb-1 mb-2">Skills</h3>
                    <div class="space-y-3">
                        {f'<div><div class="text-[10px] font-bold text-gray-700 uppercase tracking-wide mb-1">Languages</div><div class="flex flex-wrap gap-1">{render_skills_badges(data["skills_languages"])}</div></div>' if data.get("skills_languages") else ''}
                        {f'<div><div class="text-[10px] font-bold text-gray-700 uppercase tracking-wide mb-1">Tools & Tech</div><div class="flex flex-wrap gap-1">{render_skills_badges(data["skills_tools"])}</div></div>' if data.get("skills_tools") else ''}
                        {f'<div><div class="text-[10px] font-bold text-gray-700 uppercase tracking-wide mb-1">Soft Skills</div><div class="flex flex-wrap gap-1">{render_skills_badges(data["skills_soft"])}</div></div>' if data.get("skills_soft") else ''}
                    </div>
                </div>
                ''' or ''}
                
                <!-- Education stack -->
                {f'''
                <div>
                    <h3 class="text-xs font-bold text-[var(--primary-color)] uppercase tracking-widest border-b border-gray-200 pb-1 mb-2">Education</h3>
                    <div class="space-y-3">
                        {"".join([f'<div class="text-xs"><div class="font-bold text-gray-900">{edu["title"]}</div><div class="text-gray-600">{edu["subtitle"]}</div><div class="text-gray-400 font-mono mt-0.5">{edu["dates"]}</div></div>' for edu in data["education_parsed"]])}
                    </div>
                </div>
                ''' if data.get("education_parsed") else ''}
            </div>
            
            <!-- Main Column -->
            <div class="col-span-8 space-y-5">
                {f'''
                <div>
                    <h3 class="text-xs font-bold text-[var(--primary-color)] uppercase tracking-widest border-b border-gray-200 pb-1 mb-2">Summary</h3>
                    <p class="text-justify text-sm text-gray-700">{_format_markdown_to_html(data["summary"])}</p>
                </div>
                ''' if data.get("summary") else ''}
                
                {f'''
                <div>
                    <h3 class="text-xs font-bold text-[var(--primary-color)] uppercase tracking-widest border-b border-gray-200 pb-1 mb-2">Experience</h3>
                    {render_entries_html(data["experience_parsed"], "Modern Two-Column")}
                </div>
                ''' if data.get("experience_parsed") else ''}
                
                {f'''
                <div>
                    <h3 class="text-xs font-bold text-[var(--primary-color)] uppercase tracking-widest border-b border-gray-200 pb-1 mb-2">Projects & Certifications</h3>
                    {render_entries_html(data["certs_projects_parsed"], "Modern Two-Column")}
                </div>
                ''' if data.get("certs_projects_parsed") else ''}
                
                {f'''
                <div>
                    <h3 class="text-xs font-bold text-[var(--primary-color)] uppercase tracking-widest border-b border-gray-200 pb-1 mb-2">Achievements</h3>
                    {render_entries_html(data["achievements_parsed"], "Modern Two-Column")}
                </div>
                ''' if data.get("achievements_parsed") else ''}
            </div>
        </div>
        """

    elif template_style == "Vivid":
        # Two-tone name header
        first_space = name.find(" ")
        if first_space == -1:
            name_html = f'<span class="text-[var(--primary-color)] font-extrabold">{name}</span>'
        else:
            name_html = f'<span class="text-[var(--primary-color)] font-extrabold">{name[:first_space]}</span> <span class="text-[var(--primary-color)] font-light opacity-80">{name[first_space+1:]}</span>'
            
        body_html += f"""
        <div class="mb-6">
            <h1 class="text-3xl tracking-tight uppercase">{name_html}</h1>
            {f'<div class="text-sm font-mono text-gray-500 uppercase tracking-widest mt-1">{predicted_role}</div>' if predicted_role else ''}
            <div class="flex flex-wrap gap-x-4 gap-y-1.5 mt-3 text-xs font-mono">
                {f'<span class="flex items-center gap-1.5"><span class="flex items-center justify-center w-5 h-5 rounded-full border border-gray-300 text-[10px] text-gray-500"><i class="fa fa-envelope"></i></span> <a href="mailto:{contact["email"]}">{contact["email"]}</a></span>' if contact.get("email") else ''}
                {f'<span class="flex items-center gap-1.5"><span class="flex items-center justify-center w-5 h-5 rounded-full border border-gray-300 text-[10px] text-gray-500"><i class="fa fa-phone"></i></span> {contact["phone"]}</span>' if contact.get("phone") else ''}
                {f'<span class="flex items-center gap-1.5"><span class="flex items-center justify-center w-5 h-5 rounded-full border border-gray-300 text-[10px] text-gray-500"><i class="fa fa-map-marker-alt"></i></span> {contact["location"]}</span>' if contact.get("location") else ''}
                {f'<span class="flex items-center gap-1.5"><span class="flex items-center justify-center w-5 h-5 rounded-full border border-gray-300 text-[10px] text-gray-500"><i class="fab fa-linkedin"></i></span> <a href="https://{contact["linkedin"]}" target="_blank">LinkedIn</a></span>' if contact.get("linkedin") else ''}
                {f'<span class="flex items-center gap-1.5"><span class="flex items-center justify-center w-5 h-5 rounded-full border border-gray-300 text-[10px] text-gray-500"><i class="fab fa-github"></i></span> <a href="https://{contact["github"]}" target="_blank">GitHub</a></span>' if contact.get("github") else ''}
                {f'<span class="flex items-center gap-1.5"><span class="flex items-center justify-center w-5 h-5 rounded-full border border-gray-300 text-[10px] text-gray-500"><i class="fa fa-globe"></i></span> <a href="https://{contact["website"]}" target="_blank">Website</a></span>' if contact.get("website") else ''}
            </div>
        </div>
        
        <div class="grid grid-cols-12 gap-6 items-start">
            <!-- Left Main Column (63%) -->
            <div class="col-span-8 space-y-5">
                {f'''
                <div>
                    <h3 class="text-sm font-extrabold text-[var(--primary-color)] uppercase tracking-wider border-b border-gray-250 pb-1 mb-2 font-mono">Summary</h3>
                    <p class="text-justify text-sm text-gray-700">{_format_markdown_to_html(data["summary"])}</p>
                </div>
                ''' if data.get("summary") else ''}
                
                {f'''
                <div>
                    <h3 class="text-sm font-extrabold text-[var(--primary-color)] uppercase tracking-wider border-b border-gray-250 pb-1 mb-2 font-mono">Experience</h3>
                    {render_entries_html(data["experience_parsed"], "Vivid")}
                </div>
                ''' if data.get("experience_parsed") else ''}
                
                {f'''
                <div>
                    <h3 class="text-sm font-extrabold text-[var(--primary-color)] uppercase tracking-wider border-b border-gray-250 pb-1 mb-2 font-mono">Projects</h3>
                    {render_entries_html(data["certs_projects_parsed"], "Vivid")}
                </div>
                ''' if data.get("certs_projects_parsed") else ''}
            </div>
            
            <!-- Right Sidebar Column (37%) -->
            <div class="col-span-4 space-y-5">
                <!-- Skills stack -->
                {(data.get("skills_languages") or data.get("skills_tools") or data.get("skills_soft")) and f'''
                <div>
                    <h3 class="text-sm font-extrabold text-[var(--primary-color)] uppercase tracking-wider border-b border-gray-250 pb-1 mb-2 font-mono">Skills</h3>
                    <div class="space-y-3">
                        {f'<div><div class="text-[10px] font-bold text-gray-800 uppercase font-mono tracking-wide mb-1">Languages</div><div class="flex flex-wrap gap-1">{render_skills_badges(data["skills_languages"])}</div></div>' if data.get("skills_languages") else ''}
                        {f'<div><div class="text-[10px] font-bold text-gray-800 uppercase font-mono tracking-wide mb-1">Tools & Tech</div><div class="flex flex-wrap gap-1">{render_skills_badges(data["skills_tools"])}</div></div>' if data.get("skills_tools") else ''}
                        {f'<div><div class="text-[10px] font-bold text-gray-800 uppercase font-mono tracking-wide mb-1">Soft Skills</div><div class="flex flex-wrap gap-1">{render_skills_badges(data["skills_soft"])}</div></div>' if data.get("skills_soft") else ''}
                    </div>
                </div>
                ''' or ''}
                
                {f'''
                <div>
                    <h3 class="text-sm font-extrabold text-[var(--primary-color)] uppercase tracking-wider border-b border-gray-250 pb-1 mb-2 font-mono">Education</h3>
                    <div class="space-y-3">
                        {"".join([f'<div><div class="font-bold text-sm text-gray-900">{edu["title"]}</div><div class="text-xs text-gray-600">{edu["subtitle"]}</div><div class="text-[10px] text-gray-400 font-mono mt-0.5">{edu["dates"]}</div></div>' for edu in data["education_parsed"]])}
                    </div>
                </div>
                ''' if data.get("education_parsed") else ''}
                
                {f'''
                <div>
                    <h3 class="text-sm font-extrabold text-[var(--primary-color)] uppercase tracking-wider border-b border-gray-250 pb-1 mb-2 font-mono">Achievements</h3>
                    {render_entries_html(data["achievements_parsed"], "Vivid")}
                </div>
                ''' if data.get("achievements_parsed") else ''}
            </div>
        </div>
        """

    else: # Classic (Serif layout)
        body_html += f"""
        <div class="text-center mb-6">
            <h1 class="text-4xl font-bold tracking-tight text-gray-950 uppercase mb-1" style="font-family: 'Playfair Display', Georgia, serif;">{name}</h1>
            {f'<div class="text-xs tracking-widest text-gray-500 uppercase font-medium">{predicted_role}</div>' if predicted_role else ''}
            <div class="flex flex-wrap justify-center items-center gap-x-2 gap-y-0.5 mt-2.5 text-xs text-gray-700 font-serif">
                {contact["location"] if contact.get("location") else ''}
                {f' <span class="text-gray-300">•</span> ' + contact["phone"] if contact.get("phone") else ''}
                {f' <span class="text-gray-300">•</span> <a href="mailto:{contact["email"]}">{contact["email"]}</a>' if contact.get("email") else ''}
                {f' <span class="text-gray-300">•</span> <a href="https://{contact["linkedin"]}" target="_blank">LinkedIn</a>' if contact.get("linkedin") else ''}
                {f' <span class="text-gray-300">•</span> <a href="https://{contact["github"]}" target="_blank">GitHub</a>' if contact.get("github") else ''}
                {f' <span class="text-gray-300">•</span> <a href="https://{contact["website"]}" target="_blank">Website</a>' if contact.get("website") else ''}
            </div>
        </div>
        """
        
        # Sections
        if data.get("summary"):
            formatted_summary = _format_markdown_to_html(data["summary"])
            body_html += f"""
            <div class="mb-5 break-inside-avoid">
                <h2 class="section-title" style="border-bottom-width: 1px; font-family: 'Playfair Display', Georgia, serif;">Professional Summary</h2>
                <p class="text-justify text-sm text-gray-800 leading-relaxed font-serif" style="font-family: inherit;">{formatted_summary}</p>
            </div>
            """
            
        if data.get("skills_languages") or data.get("skills_tools") or data.get("skills_soft"):
            body_html += f"""
            <div class="mb-5 break-inside-avoid">
                <h2 class="section-title" style="border-bottom-width: 1px; font-family: 'Playfair Display', Georgia, serif;">Skills</h2>
                <div class="text-sm text-gray-850 font-serif space-y-1" style="font-family: inherit;">
                    {f'<div><span class="font-bold">Languages:</span> {data["skills_languages"]}</div>' if data.get("skills_languages") else ''}
                    {f'<div><span class="font-bold">Tools & Technologies:</span> {data["skills_tools"]}</div>' if data.get("skills_tools") else ''}
                    {f'<div><span class="font-bold">Soft Skills:</span> {data["skills_soft"]}</div>' if data.get("skills_soft") else ''}
                </div>
            </div>
            """
            
        if data.get("experience_parsed"):
            body_html += f"""
            <div class="mb-5">
                <h2 class="section-title" style="border-bottom-width: 1px; font-family: 'Playfair Display', Georgia, serif;">Work Experience</h2>
                {render_entries_html(data["experience_parsed"], "Classic")}
            </div>
            """
            
        if data.get("education_parsed"):
            body_html += f"""
            <div class="mb-5">
                <h2 class="section-title" style="border-bottom-width: 1px; font-family: 'Playfair Display', Georgia, serif;">Education</h2>
                {render_entries_html(data["education_parsed"], "Classic")}
            </div>
            """
            
        if data.get("achievements_parsed"):
            body_html += f"""
            <div class="mb-5">
                <h2 class="section-title" style="border-bottom-width: 1px; font-family: 'Playfair Display', Georgia, serif;">Achievements</h2>
                {render_entries_html(data["achievements_parsed"], "Classic")}
            </div>
            """
            
        if data.get("certs_projects_parsed"):
            body_html += f"""
            <div class="mb-5">
                <h2 class="section-title" style="border-bottom-width: 1px; font-family: 'Playfair Display', Georgia, serif;">Certifications & Projects</h2>
                {render_entries_html(data["certs_projects_parsed"], "Classic")}
            </div>
            """

    return html_start + body_html + html_end


def generate_enhanced_pdf_direct(
    name: str,
    summary: str,
    skills_text: str,
    experience: str,
    education: str,
    achievements: str,
    certs_projects: str,
    output_path: str = "enhanced_resume.pdf",
    font_family: str = "Helvetica",
    base_font_size: int = 10,
    margin: int = 15,
    primary_color_name: str = "Dark Blue",
    page_size: str = "A4",
    template_style: str = "Clean",
    predicted_role: str = "",
    original_text: str = "",
    skills_languages: str = "",
    skills_tools: str = "",
    skills_soft: str = ""
) -> str:
    """Generate a high-fidelity resume PDF from text inputs by compiling to HTML and printing via Playwright.

    This maps directly to the design aesthetics of Resume-Matcher templates.
    """
    # Parse unstructured text areas into lists of entry dictionaries
    experience_parsed = _parse_resume_section(experience)
    education_parsed = _parse_resume_section(education)
    achievements_parsed = _parse_resume_section(achievements)
    certs_projects_parsed = _parse_resume_section(certs_projects)
    
    # Extract contact info from original text if available
    contact = _extract_contact_info(original_text)
    
    # If categorized fields are empty but old skills_text has content, map to tools
    if not skills_languages and not skills_tools and not skills_soft and skills_text:
        skills_tools = skills_text
        
    # Compile resume data payload
    resume_data = {
        "name": name or "Your Name",
        "predicted_role": predicted_role or "",
        "summary": summary or "",
        "skills_languages": skills_languages or "",
        "skills_tools": skills_tools or "",
        "skills_soft": skills_soft or "",
        "experience_parsed": experience_parsed,
        "education_parsed": education_parsed,
        "achievements_parsed": achievements_parsed,
        "certs_projects_parsed": certs_projects_parsed,
        "contact": contact,
        "template_style": template_style,
        "font_family": font_family,
        "base_font_size": base_font_size,
        "primary_color_name": primary_color_name
    }
    
    # Generate HTML content
    html_content = _generate_html_resume(resume_data)
    
    # Write to a temporary file
    temp_dir = tempfile.gettempdir()
    temp_html_path = os.path.join(temp_dir, "temp_resume_render.html")
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    try:
        # Launch Playwright and render PDF
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            # Set viewport to normal size to ensure correct print media styling
            page.set_viewport_size({"width": 800, "height": 1000})
            
            # Go to the local file url
            file_url = f"file://{os.path.abspath(temp_html_path)}"
            page.goto(file_url, wait_until="load")
            
            # Wait for fonts & rendering
            page.evaluate("document.fonts.ready")
            
            # Print page to PDF with custom margin parameters
            pdf_margin = {"top": f"{margin}mm", "right": f"{margin}mm", "bottom": f"{margin}mm", "left": f"{margin}mm"}
            page.pdf(
                path=output_path,
                format=page_size,
                print_background=True,
                margin=pdf_margin
            )
            browser.close()
    finally:
        # Cleanup temp HTML file
        if os.path.exists(temp_html_path):
            try:
                os.remove(temp_html_path)
            except Exception:
                pass
                
    return os.path.abspath(output_path)



import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from text_utils import preprocess, identity_tokenizer

def load_jobs_corpus():
    path = "data/real_jobs_corpus.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df['description'].str.len() > 50]
        return df.to_dict('records')
    return []

def recommend_jobs(resume_text: str, top_n: int = 10) -> list:
    if not resume_text.strip():
        return []

    jobs = load_jobs_corpus()
    if not jobs:
        return []

    job_texts = [job["description"] for job in jobs]
    corpus = job_texts + [resume_text]

    # Preprocess corpus with unified tokenizer
    processed_corpus = [preprocess(text) for text in corpus]

    vectorizer = TfidfVectorizer(
        tokenizer=identity_tokenizer,
        preprocessor=identity_tokenizer,
        token_pattern=None,
        max_features=5000,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(processed_corpus)

    resume_vector = tfidf_matrix[-1]
    job_vectors   = tfidf_matrix[:-1]

    similarities = cosine_similarity(resume_vector, job_vectors).flatten()

    # Percentile rank within non-zero matches only
    # Avoids P95 collapse when 80%+ jobs are unrelated to resume
    nonzero_sims = similarities[similarities > 0.001]

    # To prevent clustering at 99%, we use Baseline Max-Relative Scaling.
    # We find the top score. If the top score is garbage, we use a much higher baseline (0.30) 
    # to prevent a terrible 0.08 match from artificially inflating to 56%.
    max_raw = similarities.max()
    baseline = max(max_raw, 0.30)

    ranked_indices = np.argsort(similarities)[::-1]
    total_jobs     = len(jobs)

    results = []
    for rank_pos, idx in enumerate(ranked_indices[:top_n]):
        raw       = float(similarities[idx])
        job_match = jobs[idx].copy()

        # Scale relative to the baseline. Top job gets ~96%.
        scaled_score = int((raw / baseline) * 96)
        
        # Enforce harsh penalty on objectively terrible semantic matches (< 0.10 raw)
        if raw < 0.10:
            scaled_score = min(scaled_score, 25) # Never let a terrible raw score exceed 25% match
        
        # Cap at 99 and floor at 0
        final_score = min(max(scaled_score, 0), 99)

        corpus_rank = rank_pos + 1
        corpus_pct  = int((1 - corpus_rank / total_jobs) * 100)

        match_label = (
            "Excellent" if final_score >= 80 else
            "Good"      if final_score >= 50 else
            "Fair"
        )
        
        # Explainability: Direct Dictionary Extraction (Bypass TF-IDF penalties for common tech skills)
        from skills_dictionary import SKILLS_DB
        import re
        
        job_desc_lower = str(job_match.get("description", "")).lower()
        
        # Extract every skill from the text that exists in our massive database
        extracted_job_skills = []
        for skill in SKILLS_DB:
            # Use word boundaries to prevent "r" from matching inside "manager" or "go" inside "algorithm"
            if re.search(r'\b' + re.escape(skill) + r'\b', job_desc_lower):
                extracted_job_skills.append(skill.title())
                
        resume_lower = resume_text.lower()
        resume_tokens = set(processed_corpus[-1])
        
        # We also need to check the raw resume text for multi-word skills like "machine learning"
        matched_skills = []
        missing_skills = []
        
        for skill in extracted_job_skills:
            # Check if the skill is in the resume (using regex to avoid partial matches)
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', resume_lower):
                if len(matched_skills) < 8:
                    matched_skills.append(skill)
            else:
                if len(missing_skills) < 8:
                    missing_skills.append(skill)

        job_match["match_score"] = final_score
        job_match["match_label"] = match_label
        job_match["matched_skills"] = matched_skills
        job_match["missing_skills"] = missing_skills
        job_match["percentile"]  = corpus_pct
        job_match["raw_score"]   = round(raw, 5)
        job_match["pool_size"]   = len(nonzero_sims)
        job_match["corpus_p95"]  = round(float(np.percentile(similarities, 95)), 5)

        results.append(job_match)

    return results

def get_domain_centroid_score(resume_text: str, predicted_domain: str) -> float:
    """
    Computes the Cosine Similarity between the user's resume and the
    aggregate TF-IDF centroid of all jobs in the predicted domain.
    """
    if not resume_text.strip() or not predicted_domain:
        return 0.0

    jobs = load_jobs_corpus()
    if not jobs:
        return 0.0

    # Filter jobs that belong to the predicted domain (heuristic matching for now)
    domain_keywords = predicted_domain.lower().split()
    domain_jobs = []
    for job in jobs:
        title = str(job.get('title', '')).lower()
        if any(k in title for k in domain_keywords):
            domain_jobs.append(job)
            
    # If no exact domain matches, fallback to all jobs to compute a general centroid
    if not domain_jobs:
        domain_jobs = jobs

    job_texts = [job["description"] for job in domain_jobs]
    corpus = job_texts + [resume_text]

    processed_corpus = [preprocess(text) for text in corpus]

    vectorizer = TfidfVectorizer(
        tokenizer=identity_tokenizer,
        preprocessor=identity_tokenizer,
        token_pattern=None,
        max_features=5000,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(processed_corpus)

    resume_vector = tfidf_matrix[-1]
    job_vectors = tfidf_matrix[:-1]

    # Calculate centroid of domain jobs
    centroid_vector = np.asarray(job_vectors.mean(axis=0))
    
    # Compute similarity to centroid
    similarity = cosine_similarity(resume_vector, centroid_vector).flatten()[0]
    
    # Scale to 0-100
    score = min(max(similarity * 100 * 2, 0), 100) # x2 to inflate low raw cosine values
    return round(float(score), 2)

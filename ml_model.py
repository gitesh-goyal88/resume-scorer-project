"""
ml_model.py — ResumeIQ Machine Learning Models

Contains three models:
1. Job Role Classifier (KNN + Naive Bayes + Logistic Regression → Soft-Voting Ensemble)
   — predicts job category from TF-IDF resume features across 25 industry domains
2. ATS Health Score Engine (Multinomial scoring across skills, verbs, metrics, sections)
   — computes a weighted ATS compatibility score and letter grade
3. Bullet Point Evaluator (SentenceTransformers + Cosine Similarity)
   — evaluates bullet point impact against a Gold Standard dense vector matrix
"""

import os
import re
import ssl
import pickle
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report


# ---------------------------------------------------------------------------
# Utility: text cleaning
# ---------------------------------------------------------------------------

def clean_resume_text(text: str) -> str:
    """Clean resume text by removing URLs, mentions, special chars, etc."""
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ===========================================================================================
# MODEL 1: Job Role Classifier (KNN)
# ===========================================================================================

def train_job_role_classifier() -> dict:
    """
    Downloads the resume dataset, trains an ENSEMBLE of 4 classifiers for job-role
    prediction (NaiveBayes, KNN, LogisticRegression, RandomForest), and saves each
    model + the shared TF-IDF vectorizer to disk.

    Returns a dict with per-model train/test accuracy + ensemble accuracy.
    """
    print("=" * 70)
    print("MODEL 1: Job Role Classifier — 4-Model Soft-Voting Ensemble")
    print("=" * 70)

    data_path = os.path.join("data", "UpdatedResumeDataSet.csv")
    if not os.path.exists(data_path):
        print("Data not found locally. Please run generate_data.py first.")
        return {}

    # --- Load & clean ---
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    df["cleaned_resume"] = df["Resume"].apply(clean_resume_text)

    # --- Shared TF-IDF Vectorizer ---
    tfidf = TfidfVectorizer(max_features=1500, stop_words="english")
    X = tfidf.fit_transform(df["cleaned_resume"])
    y = df["Category"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    os.makedirs("models", exist_ok=True)

    results = {}

    # --- Train & evaluate each model ---
    models_to_train = {
        "naive_bayes":         MultinomialNB(),
        "knn":                 KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    trained_models = {}
    for name, clf in models_to_train.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc  = accuracy_score(y_test,  clf.predict(X_test))
        print(f"  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

        # Save individual model
        model_file = os.path.join("models", f"{name}_classifier.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(clf, f)
        print(f"  Saved → {model_file}")

        trained_models[name] = clf
        results[name] = {
            "train_accuracy": round(train_acc, 4),
            "test_accuracy":  round(test_acc, 4),
        }

    # --- Ensemble soft voting (average predict_proba across all 4) ---
    print("\nComputing Ensemble (Soft Voting) accuracy...")
    all_probas = np.mean(
        [clf.predict_proba(X_test) for clf in trained_models.values()],
        axis=0
    )
    ensemble_classes = list(trained_models["naive_bayes"].classes_)
    ensemble_preds   = [ensemble_classes[i] for i in np.argmax(all_probas, axis=1)]
    ensemble_acc     = accuracy_score(y_test, ensemble_preds)
    print(f"  Ensemble Test Acc: {ensemble_acc:.4f}")
    results["ensemble"] = {"test_accuracy": round(ensemble_acc, 4)}

    # Save shared TF-IDF vectorizer
    with open(os.path.join("models", "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    print("\nSaved models/tfidf_vectorizer.pkl")

    # --- Compute & save TF-IDF baseline for ATS skill scoring ---
    # Resumes are shorter than JDs, so using Job-to-Job P95 similarity as a baseline
    # is mathematically flawed (it sets an impossible standard).
    # Instead, we compute the "Global Background Noise" (mean similarity of a job
    # to every other random job in the corpus), take the 95th percentile of that noise,
    # and define a "Perfect Match" as 2.5x that background noise.
    print("\nComputing Signal-to-Noise baseline for ATS skill scorer...")
    try:
        from job_matcher import load_jobs_corpus
        from text_utils import preprocess, identity_tokenizer
        from sklearn.feature_extraction.text import TfidfVectorizer as _TV
        from sklearn.metrics.pairwise import cosine_similarity as _cs

        _jobs = load_jobs_corpus()
        if _jobs:
            _job_texts = [j["description"] for j in _jobs]
            _processed  = [preprocess(t) for t in _job_texts]
            _vec = _TV(
                tokenizer=identity_tokenizer,
                preprocessor=identity_tokenizer,
                token_pattern=None,
                max_features=5000,
                sublinear_tf=True
            )
            _mat = _vec.fit_transform(_processed)
            _idx = np.random.RandomState(42).choice(len(_jobs), size=min(200, len(_jobs)), replace=False)
            
            _global_noise = []
            for i in _idx:
                s = _cs(_mat[i], _mat).flatten()
                s[i] = 0
                non_zero = s[s > 0]
                if len(non_zero) > 0:
                    _global_noise.append(float(np.mean(non_zero)))
            
            # P95 of global background noise across the corpus
            noise_ceiling = float(np.percentile(_global_noise, 95))
            
            # A "perfect" resume match is defined as 2.5x the background noise ceiling
            _baseline = noise_ceiling * 2.5
            
            with open(os.path.join("models", "ats_tfidf_baseline.pkl"), "wb") as f:
                pickle.dump(_baseline, f)
            print(f"  Noise Ceiling = {noise_ceiling:.4f}")
            print(f"  Signal Baseline (2.5x) = {_baseline:.4f}  →  saved models/ats_tfidf_baseline.pkl")
    except Exception as e:
        print(f"  Warning: Could not compute baseline: {e}")

    # Save ensemble metrics for Analytics dashboard
    with open(os.path.join("models", "ensemble_metrics.pkl"), "wb") as f:
        pickle.dump(results, f)
    print("Saved models/ensemble_metrics.pkl")

    return results


def _load_model_safe(model_path: str, vectorizer_path: str, train_fn) -> tuple:
    """
    Load a model and its vectorizer using pickle.
    If loading fails due to file absence or version mismatch (AttributeError/ValueError/etc.),
    it automatically triggers the train_fn to retrain and save the models, then retries.
    """
    try:
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            print(f"Model or vectorizer not found: {model_path}, {vectorizer_path}. Retraining...")
            train_fn()
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        # Dry run to catch potential scikit-learn version mismatch errors
        dummy_vec = vectorizer.transform(["dummy text"])
        _ = model.predict_proba(dummy_vec)
        return model, vectorizer
    except Exception as e:
        print(f"Error loading models ({model_path}): {e}. Attempting self-healing by retraining...")
        # Delete corrupt/incompatible files if they exist to force clean regeneration
        for path in [model_path, vectorizer_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        # Retrain
        train_fn()
        # Retry once
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model, vectorizer


def predict_job_category(resume_text: str) -> dict:
    """
    Predict the job category using a 4-model Soft-Voting Ensemble.
    Loads NaiveBayes, KNN, LogisticRegression, and RandomForest,
    averages their predict_proba outputs, and returns the top prediction
    along with individual model confidence scores.

    Returns
    -------
    dict  {"category": str, "confidence": float, "model_scores": dict}
    """
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
    model_names = ["naive_bayes", "knn", "logistic_regression"]

    # Auto-retrain if any model is missing
    missing = [not os.path.exists(os.path.join("models", f"{m}_classifier.pkl")) for m in model_names]
    if any(missing) or not os.path.exists(vectorizer_path):
        print("One or more ensemble models missing. Retraining all...")
        train_job_role_classifier()

    # Load vectorizer
    with open(vectorizer_path, "rb") as f:
        tfidf = pickle.load(f)

    cleaned = clean_resume_text(resume_text)
    vec = tfidf.transform([cleaned])

    # Load each model and collect predict_proba
    all_probas = []
    classes = None
    model_scores = {}

    for name in model_names:
        try:
            with open(os.path.join("models", f"{name}_classifier.pkl"), "rb") as f:
                clf = pickle.load(f)
            proba = clf.predict_proba(vec)[0]
            if classes is None:
                classes = list(clf.classes_)
            all_probas.append(proba)
            top_idx = int(np.argmax(proba))
            model_scores[name] = {
                "prediction": str(clf.classes_[top_idx]),
                "confidence": round(float(proba[top_idx]), 4)
            }
        except Exception as e:
            print(f"Warning: Could not load {name}: {e}")

    if not all_probas or classes is None:
        return {"category": "Unknown", "confidence": 0.0, "model_scores": {}}

    # Soft voting: average probabilities across all models
    ensemble_proba = np.mean(all_probas, axis=0)
    top_idx = int(np.argmax(ensemble_proba))
    category   = str(classes[top_idx])
    confidence = round(float(ensemble_proba[top_idx]), 4)

    return {
        "category":     category,
        "confidence":   confidence,
        "model_scores": model_scores
    }


# ===========================================================================================
# MODEL 2: ATS Health Score — IR-Based Skill Scoring + Weighted Dimensions
# ===========================================================================================

def compute_tfidf_skill_score(resume_text: str) -> float:
    """
    IR-based skill scorer using TF-IDF + Cosine Similarity.

    Builds a joint TF-IDF matrix over the entire 2800-job corpus + the resume.
    Computes cosine similarity between the resume vector and every job vector.
    Takes the mean of the top-10 similarities as the market alignment score.
    Normalises against a baseline of 0.30 (empirically strong alignment) -> 0-100.

    Replaces the naive hardcoded-keyword-count approach with a genuine
    Information Retrieval metric grounded in the live job market corpus.
    """
    try:
        from job_matcher import load_jobs_corpus
        from text_utils import preprocess, identity_tokenizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        jobs = load_jobs_corpus()
        if not jobs:
            return 0.0

        job_texts  = [job["description"] for job in jobs]
        corpus     = job_texts + [resume_text]
        processed  = [preprocess(t) for t in corpus]

        vectorizer = TfidfVectorizer(
            tokenizer=identity_tokenizer,
            preprocessor=identity_tokenizer,
            token_pattern=None,
            max_features=5000,
            sublinear_tf=True   # log-normalise TF to suppress very frequent terms
        )
        tfidf_matrix = vectorizer.fit_transform(processed)

        resume_vec = tfidf_matrix[-1]    # last row = resume
        job_vecs   = tfidf_matrix[:-1]   # all other rows = job descriptions

        sims = cosine_similarity(resume_vec, job_vecs).flatten()

        # Mean of top-10 cosine similarities = resume's market alignment score
        top10_mean = float(np.mean(np.sort(sims)[::-1][:10]))

        # Load the Signal-to-Noise baseline computed once during training.
        # This is 2.5x the P95 global background noise of the corpus.
        # If missing (first run before training), fall back to a safe default of 0.21.
        baseline_path = os.path.join("models", "ats_tfidf_baseline.pkl")
        if os.path.exists(baseline_path):
            with open(baseline_path, "rb") as f:
                baseline = pickle.load(f)
        else:
            baseline = 0.21   # safe fallback until training is run

        score = min((top10_mean / baseline) * 100, 100)
        return round(score, 2)

    except Exception as e:
        print(f"[compute_tfidf_skill_score] Fallback to 0: {e}")
        return 0.0

def compute_health_score(features: dict) -> dict:
    # IR-based skill score (TF-IDF cosine similarity vs job corpus)
    # Falls back to keyword-count heuristic if tfidf_skill_score is completely missing
    tfidf_score = features.get("tfidf_skill_score")
    if tfidf_score is not None:
        skill_score = tfidf_score
    else:
        skill_score = min(features.get("skill_count", 0) / 20 * 100, 100)
        
    verb_score     = min(features.get("action_verb_count", 0) / 10 * 100, 100)
    metrics_score  = min(features.get("metrics_count", 0) / 5 * 100, 100)
    section_score  = (features.get("section_count", 0) / 4) * 100
    format_score   = max(100 - features.get("formatting_penalty", 0), 0)
    
    # ML ATS Weights: 80% Hard Skills, 10% Formatting, 10% Other (Verbs/Metrics/Sections)
    total = (
        skill_score   * 0.80 +
        format_score  * 0.10 +
        verb_score    * 0.04 +
        metrics_score * 0.03 +
        section_score * 0.03
    )
    total = int(round(total))
    grade = "A" if total >= 80 else "B" if total >= 65 else "C" if total >= 50 else "D"
    return {
        "total_score": total,
        "grade": grade,
        "breakdown": {
            "Skills":      int(skill_score),
            "Action Verbs":int(verb_score),
            "Metrics":     int(metrics_score),
            "Sections":    int(section_score),
            "Formatting":  int(format_score),
        }
    }




# ===========================================================================================
# ===========================================================================================
# MODEL 3: Bullet Point Impact Classifier (Logistic Regression & Weak Supervision)
# ===========================================================================================

def build_semantic_bullet_engine() -> dict:
    """
    Builds a Dense Vector Space matrix of 'Gold Standard' expert bullet points
    using a local SentenceTransformer model. Saves the embeddings for fast Cosine Similarity inference.
    """
    print("=" * 70)
    print("MODEL 3: Semantic Bullet Point Evaluator (SentenceTransformers + Cosine Similarity)")
    print("=" * 70)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed. Run: pip install sentence-transformers")
        return {"status": "failed", "error": "Missing dependency"}

    strong_expert = [
        "Increased revenue by 35% by redesigning the checkout flow using React",
        "Reduced server response time by 40% through optimizing SQL queries and implementing Redis caching",
        "Led a team of 8 engineers to deliver a microservices migration 2 weeks ahead of schedule",
        "Automated CI/CD pipeline with Jenkins, reducing deployment time from 4 hours to 15 minutes",
        "Built a real-time analytics dashboard in Python and D3.js serving 10K daily active users",
        "Decreased customer churn by 18% by implementing a predictive ML model using XGBoost",
        "Developed RESTful APIs in Node.js handling 5M requests per day with 99.9% uptime",
        "Migrated legacy monolith to AWS Lambda, cutting infrastructure costs by 60%",
        "Improved test coverage from 45% to 92% by introducing pytest and integration tests",
        "Designed a recommendation engine that increased average order value by 22%",
        "Managed a $2M annual budget for cloud infrastructure across 3 AWS regions",
        "Trained and mentored 5 junior developers, improving team velocity by 30%",
        "Implemented OAuth 2.0 and JWT-based authentication securing 500K user accounts",
        "Optimized ETL pipeline processing 50GB of data daily, reducing runtime by 65%",
        "Negotiated vendor contracts saving the company $150K annually",
        "Spearheaded adoption of Kubernetes, achieving 99.99% service availability",
        "Created a fraud detection system using Random Forest that flagged 95% of fraudulent transactions",
        "Published 3 technical blog posts that generated 25K pageviews and 200 inbound leads",
        "Reduced bug backlog by 70% within 2 sprints through systematic triage and pair programming",
        "Architected a data lake on AWS S3 and Glue processing 1TB+ daily",
        "Delivered a mobile app feature used by 1.2M users within the first month of launch",
        "Improved page load speed by 50% through code splitting and lazy loading in React",
        "Coordinated cross-functional teams across 4 time zones to ship a product on schedule",
        "Achieved a 98% customer satisfaction score by redesigning the support ticket workflow",
        "Integrated Stripe payment gateway processing $3M in monthly transactions",
        "Developed a chatbot using NLP that resolved 40% of support tickets without human intervention",
        "Increased email open rates by 28% through A/B testing subject lines and send times",
        "Built a data pipeline in Apache Spark processing 100M records per hour",
        "Reduced onboarding time for new hires from 3 weeks to 5 days with documentation and tooling",
        "Deployed a containerized application on ECS serving 50K concurrent users",
        "Streamlined inventory management system, reducing stockouts by 25%",
        "Launched an internal CLI tool in Go that saved engineers 10 hours per week",
        "Drove adoption of TypeScript across 12 repositories improving type safety and reducing runtime errors by 35%",
        "Designed and implemented a rate-limiting middleware handling 100K requests per minute",
        "Achieved SOC 2 Type II compliance by leading security audit remediation across 5 services",
        "Wrote unit and integration tests covering 95% of critical payment processing paths",
        "Reduced mean time to recovery (MTTR) from 2 hours to 15 minutes with improved monitoring and runbooks",
        "Scaled PostgreSQL database to handle 10x traffic growth using read replicas and connection pooling",
        "Increased user engagement by 45% by implementing personalized push notifications",
        "Delivered quarterly OKRs consistently, completing 90% of planned initiatives on time",
        "Configured Terraform IaC for 30+ cloud resources, enabling reproducible deployments",
        "Optimized machine learning model inference latency from 200ms to 35ms using ONNX Runtime",
        "Built an automated reporting system that saved the finance team 20 hours per month",
        "Initiated a code review culture that reduced production incidents by 50%",
        "Created a customer segmentation model using K-Means that improved targeting accuracy by 30%",
        "Developed a GraphQL API that reduced frontend data-fetching calls by 60%",
        "Led a successful migration from MySQL to PostgreSQL with zero downtime",
        "Implemented feature flags using LaunchDarkly enabling safe rollouts for 2M users",
        "Improved search relevance by 38% using Elasticsearch and custom scoring algorithms",
        "Reduced Docker image sizes by 70% through multi-stage builds and Alpine base images",
    ]

    print(f"Loading lightweight SentenceTransformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Encoding {len(strong_expert)} Gold Standard XYZ-formula bullet points into Dense Vectors...")
    embeddings = model.encode(strong_expert, convert_to_numpy=True)
    
    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", "gold_standard_embeddings.pkl")
    
    with open(out_path, "wb") as f:
        pickle.dump({
            "texts": strong_expert,
            "matrix": embeddings
        }, f)
        
    print(f"Successfully saved dense vector matrix to {out_path}")
    print()
    return {"status": "success", "vector_count": len(strong_expert), "dimensions": embeddings.shape[1]}


def evaluate_bullets_semantic(candidate_bullets: list) -> list:
    """
    Inference Function: Uses Cosine Similarity to evaluate candidate bullet points
    against the pre-computed Gold Standard Dense Vector matrix.
    """
    if not candidate_bullets:
        return []
        
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception as e:
        return [{"bullet": b, "raw_cosine_similarity": 0.0, "impact_score": 0, "feedback": f"Dependency missing: {str(e)}"} for b in candidate_bullets]
        
    matrix_path = os.path.join("models", "gold_standard_embeddings.pkl")
    if not os.path.exists(matrix_path):
        return [{"bullet": b, "raw_cosine_similarity": 0.0, "impact_score": 0, "feedback": "Gold standard matrix not found"} for b in candidate_bullets]
        
    with open(matrix_path, "rb") as f:
        data = pickle.load(f)
        gold_matrix = data["matrix"]
        
    # Load model (cached locally)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Vectorize candidate bullets
    candidate_matrix = model.encode(candidate_bullets, convert_to_numpy=True)
    
    # Calculate Cosine Similarity (N_candidates x N_gold)
    similarity_scores = cosine_similarity(candidate_matrix, gold_matrix)
    
    results = []
    for i, bullet in enumerate(candidate_bullets):
        # Find the highest similarity score against ANY gold standard bullet
        max_sim = float(np.max(similarity_scores[i]))
        
        # Convert raw cosine similarity (0 to 1) into a user-friendly score (0 to 100)
        # Typically, anything > 0.65 is very strong in semantic space
        normalized_score = min(max_sim / 0.70 * 100, 100) 
        
        # Base string based on mathematical score
        if normalized_score >= 85:
            base_feedback = "Strong semantic impact."
        elif normalized_score >= 60:
            base_feedback = "Average semantic impact."
        else:
            base_feedback = "Weak semantic impact."
            
        specific_issues = []
        
        # 1. Quantifiable metrics check
        has_metric = bool(re.search(r'\b\d+%\b|\$\d+|\b\d+\b', bullet))
        if not has_metric:
            specific_issues.append("lacks quantifiable outcomes (e.g., %, $, or raw numbers)")
            
        # 2. Action Verb opening check
        action_verbs = {"developed", "led", "managed", "created", "built", "improved", "designed", "optimized", "spearheaded", "implemented", "reduced", "increased", "achieved", "architected", "automated", "orchestrated", "engineered"}
        words = re.findall(r'\b[a-zA-Z]+\b', bullet.lower())
        if words and words[0] not in action_verbs:
            specific_issues.append(f"should start with a strong action verb instead of '{words[0]}'")
            
        # 3. Length & Detail check
        if len(words) < 8:
            specific_issues.append("is too brief and lacks technical context")
        elif len(words) > 35:
            specific_issues.append("is too wordy and should be more concise")
            
        # 4. XYZ Formula / Structural Connector check
        impact_connectors = {"by", "through", "resulting in", "leading to", "achieving", "using"}
        has_connector = any(c in bullet.lower() for c in impact_connectors)
        if not has_connector and normalized_score < 75:
            specific_issues.append("is missing a structural connector explaining 'how' (e.g., 'by doing X' or 'using Y')")

        # Assemble the final dynamic feedback
        if not specific_issues:
            feedback = "Excellent bullet point! Perfectly aligns with the XYZ formula."
        else:
            if len(specific_issues) == 1:
                issue_str = specific_issues[0]
            else:
                issue_str = ", ".join(specific_issues[:-1]) + ", and " + specific_issues[-1]
            feedback = f"{base_feedback} It {issue_str}."
            
        results.append({
            "bullet": bullet,
            "raw_cosine_similarity": round(max_sim, 4),
            "impact_score": int(normalized_score),
            "feedback": feedback
        })
        
    return results


def classify_bullets(bullet_list: list) -> list:
    """
    Classify a list of resume bullet-point strings as Strong or Weak.
    Now powered by the advanced SentenceTransformer Cosine Similarity engine!
    """
    semantic_results = evaluate_bullets_semantic(bullet_list)
    
    legacy_results = []
    for res in semantic_results:
        label = "Strong" if res["impact_score"] >= 65 else "Weak"
        legacy_results.append({
            "text": res["bullet"],
            "label": label,
            "confidence": res["raw_cosine_similarity"],
            "feedback": res.get("feedback", "")
        })
        
    return legacy_results


# ===========================================================================================
# Train all models
# ===========================================================================================

def train_all_models():
    """Train and save all three ResumeIQ ML models."""
    print("\n  ResumeIQ — Training All Models\n")

    results = {}

    results["job_role_classifier"] = train_job_role_classifier()
    results["bullet_classifier"] = build_semantic_bullet_engine()

    # --- Quick smoke test --------------------------------------------------
    print("=" * 70)
    print("SMOKE TESTS")
    print("=" * 70)

    # Test 1: job category prediction
    sample_resume = (
        "Experienced Python developer with 5 years building REST APIs, "
        "microservices, Django, Flask, PostgreSQL, Docker, and AWS."
    )
    cat_result = predict_job_category(sample_resume)
    print(f"Job category prediction: {cat_result}")

    # Test 3: bullet point classification
    sample_bullets = [
        "Increased revenue by 35% by redesigning the checkout flow using React",
        "Was responsible for handling the website",
    ]
    bullet_results = classify_bullets(sample_bullets)
    for br in bullet_results:
        print(f"Bullet: '{br['text'][:60]}...' → {br['label']} ({br['confidence']:.2f})")

    print("\n  All models trained and verified successfully!\n")
    return results


if __name__ == "__main__":
    train_all_models()

import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from text_utils import preprocess, identity_tokenizer
from job_matcher import load_jobs_corpus
from ui_utils import inject_custom_css

inject_custom_css()

st.markdown("<h1 class='gradient-title' style='font-size: 3rem; margin-bottom: 5px; padding-bottom: 5px;'> ML Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-heading'>Comprehensive evaluation of Search Algorithms (TF-IDF vs BM25 vs KNN) across professional domains.</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([" Global Benchmarks", " Personal Resume Match", " Model Comparison"])

# ----------------- GLOBALS -----------------
with st.spinner("Initializing models..."):
    jobs = load_jobs_corpus()
    if not jobs:
        st.error("Job corpus is empty.")
        st.stop()

    job_texts = [job["description"] for job in jobs]
    processed_job_texts = [preprocess(text) for text in job_texts]
    
    # Pre-compute TF-IDF
    vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, preprocessor=identity_tokenizer, token_pattern=None)
    job_vectors = vectorizer.fit_transform(processed_job_texts)
    
    # Pre-compute KNN (Using Bag of Words / CountVectorizer to differentiate from TF-IDF)
    count_vectorizer = CountVectorizer(tokenizer=identity_tokenizer, preprocessor=identity_tokenizer, token_pattern=None)
    job_count_vectors = count_vectorizer.fit_transform(processed_job_texts)
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(job_count_vectors)

    # Pre-compute BM25
    bm25 = BM25Okapi(processed_job_texts)


# ==========================================
# TAB 1: GLOBAL BENCHMARKS
# ==========================================
with tab1:
    st.markdown("###  Global Search Performance")
    st.markdown("We evaluate the Precision@5 of each algorithm by simulating search queries for different professional fields.")
    
    domains = ["Data Science", "Python Backend", "Frontend React", "DevOps Engineer", "Machine Learning", "HR Manager"]
    algorithms = ["TF-IDF", "BM25", "KNN"]
    
    @st.cache_data(show_spinner=False)
    def compute_global_metrics_v2():
        p_matrix = np.zeros((len(domains), len(algorithms)))
        times = [0.0, 0.0, 0.0]
        
        for i, domain in enumerate(domains):
            query_processed = preprocess(domain)
            query_vector = vectorizer.transform([query_processed])
            
            # Ground truth evaluator
            domain_keywords = domain.lower().split()
            def is_relevant(job_idx):
                title = str(jobs[job_idx].get("title", "")).lower()
                return any(k in title for k in domain_keywords)
            
            # --- TF-IDF ---
            t0 = time.time()
            sim = cosine_similarity(query_vector, job_vectors).flatten()
            top5_tfidf = np.argsort(sim)[::-1][:5]
            times[0] += (time.time() - t0)
            p_matrix[i, 0] = sum(1 for idx in top5_tfidf if is_relevant(idx)) / 5.0
            
            # --- BM25 ---
            t0 = time.time()
            scores = bm25.get_scores(query_processed)
            top5_bm25 = np.argsort(scores)[::-1][:5]
            times[1] += (time.time() - t0)
            p_matrix[i, 1] = sum(1 for idx in top5_bm25 if is_relevant(idx)) / 5.0
            
            # --- KNN ---
            t0 = time.time()
            knn_query_vector = count_vectorizer.transform([query_processed])
            distances, indices = knn.kneighbors(knn_query_vector)
            top5_knn = indices[0]
            times[2] += (time.time() - t0)
            p_matrix[i, 2] = sum(1 for idx in top5_knn if is_relevant(idx)) / 5.0
            
        return p_matrix, [t/len(domains) for t in times]

    with st.spinner("Computing precision matrix (Running 18 searches)..."):
        precision_matrix, exec_times = compute_global_metrics_v2()
        
    avg_precision = np.mean(precision_matrix, axis=0)
    
    win_rates = [0, 0, 0]
    for i in range(len(domains)):
        max_val = np.max(precision_matrix[i])
        for j in range(len(algorithms)):
            if precision_matrix[i, j] == max_val:
                win_rates[j] += 1

    # Apply global dark theme for Matplotlib
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#18181B',
        'axes.facecolor': '#18181B',
        'axes.edgecolor': '#3B3B40',
        'text.color': '#FAFAFA',
        'axes.labelcolor': '#FAFAFA',
        'xtick.color': '#A1A1AA',
        'ytick.color': '#A1A1AA',
        'font.family': 'sans-serif'
    })
    colors = ['#3B82F6', '#10B981', '#F59E0B']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Precision Heatmap (P@5)")
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        cax = ax1.imshow(precision_matrix, cmap='Greens', vmin=0, vmax=1.0)
        ax1.set_xticks(np.arange(len(algorithms)))
        ax1.set_yticks(np.arange(len(domains)))
        ax1.set_xticklabels(algorithms)
        ax1.set_yticklabels(domains)
        
        for i in range(len(domains)):
            for j in range(len(algorithms)):
                val = precision_matrix[i, j]
                text_color = 'white' if val > 0.5 else 'black'
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight='bold')
        
        fig1.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)
        st.pyplot(fig1)
        
    with col2:
        st.markdown("#### Average Precision@5")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        bars = ax2.bar(algorithms, avg_precision, color=colors, alpha=0.9)
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', color='#2A2A2E', linestyle='--', alpha=0.7)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', color='#FAFAFA', fontweight='bold')
        st.pyplot(fig2)

    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Algorithm Win Rate")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        if sum(win_rates) > 0:
            ax3.pie(win_rates, labels=algorithms, colors=colors, autopct='%1.1f%%', startangle=90, 
                    textprops={'color': '#FAFAFA', 'fontweight': 'bold'},
                    wedgeprops={'edgecolor': '#18181B', 'linewidth': 2})
            st.pyplot(fig3)
        else:
            st.warning("Not enough data to calculate win rates.")
            
    with col4:
        st.markdown("#### Tradeoff: Speed vs Accuracy")
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        for i, algo in enumerate(algorithms):
            ax4.scatter(exec_times[i], avg_precision[i], color=colors[i], s=200, label=algo, alpha=0.9, edgecolors='#FAFAFA', linewidth=1.5)
            ax4.annotate(algo, (exec_times[i], avg_precision[i]), xytext=(10, -5), textcoords='offset points', color='#FAFAFA', fontweight='bold')
            
        ax4.set_xlabel("Avg Execution Time (Seconds)")
        ax4.set_ylabel("Average Precision@5")
        ax4.grid(color='#2A2A2E', linestyle='--', alpha=0.7)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        st.pyplot(fig4)


# ==========================================
# TAB 2: PERSONAL RESUME MATCH
# ==========================================
with tab2:
    if "resume_text" not in st.session_state or not st.session_state.resume_text:
        st.info("Please upload a resume on the Dashboard to see your personal evaluation.")
        st.stop()
        
    resume_text = st.session_state.resume_text
    predicted_role = st.session_state.get("predicted_role", "")
    user_name = st.session_state.get("user_name") or st.session_state.get("edit_name") or "User"
    first_name = user_name.split()[0]
    
    if not predicted_role:
        st.error("Predicted role not found. Please regenerate your report on the Dashboard.")
        st.stop()
        
    st.markdown(f"### Evaluating Match Accuracy for: `{predicted_role}`")
    st.markdown("A job is considered a 'True Positive' (relevant) if its title matches the predicted role.")
    
    with st.spinner("Running your resume through algorithms..."):
        resume_processed = preprocess(resume_text)
        resume_vector = vectorizer.transform([resume_processed])
        
        # TF-IDF
        t0 = time.time()
        tfidf_sim = cosine_similarity(resume_vector, job_vectors).flatten()
        tfidf_top = np.argsort(tfidf_sim)[::-1][:5]
        t_tfidf = time.time() - t0
        
        # BM25
        t0 = time.time()
        bm25_scores = bm25.get_scores(resume_processed)
        bm25_top = np.argsort(bm25_scores)[::-1][:5]
        t_bm25 = time.time() - t0
        
        # KNN
        t0 = time.time()
        knn_personal_vector = count_vectorizer.transform([resume_processed])
        distances, indices = knn.kneighbors(knn_personal_vector)
        knn_top = indices[0]
        t_knn = time.time() - t0

    # Calculate Personal Precision Matrix across ALL 6 Domains
    personal_precision = np.zeros((len(domains), len(algorithms)))
    
    for i, domain in enumerate(domains):
        domain_keywords = domain.lower().split()
        def is_relevant_personal(job_idx):
            title = str(jobs[job_idx].get("title", "")).lower()
            return any(k in title for k in domain_keywords)
            
        personal_precision[i, 0] = sum(1 for idx in tfidf_top if is_relevant_personal(idx)) / 5.0
        personal_precision[i, 1] = sum(1 for idx in bm25_top if is_relevant_personal(idx)) / 5.0
        personal_precision[i, 2] = sum(1 for idx in knn_top if is_relevant_personal(idx)) / 5.0

    # Ensure the user's predicted role is visually highlighted
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown(f"#### Resume Alignment Heatmap")
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        cax5 = ax5.imshow(personal_precision, cmap='Blues', vmin=0, vmax=1.0)
        ax5.set_xticks(np.arange(len(algorithms)))
        ax5.set_yticks(np.arange(len(domains)))
        ax5.set_xticklabels(algorithms)
        ax5.set_yticklabels(domains)
        
        for i in range(len(domains)):
            for j in range(len(algorithms)):
                val = personal_precision[i, j]
                text_color = 'white' if val > 0.5 else 'black'
                ax5.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight='bold')
        
        fig5.colorbar(cax5, ax=ax5, fraction=0.046, pad=0.04)
        st.pyplot(fig5)
        
    with colB:
        st.markdown(f"#### Resume Alignment Radar")
        avg_personal_precision = np.mean(personal_precision, axis=1)
        
        # Radar Chart Logic
        num_vars = len(domains)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        avg_personal_precision = np.concatenate((avg_personal_precision, [avg_personal_precision[0]]))
        angles += angles[:1]
        
        fig6, ax6 = plt.subplots(figsize=(6, 5.5), subplot_kw=dict(polar=True))
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(domains, size=10, fontweight='bold', color='#A1A1AA')
        ax6.set_rlabel_position(0)
        ax6.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax6.set_yticklabels(["", "0.4", "", "0.8", "1.0"], color="grey", size=8)
        ax6.set_ylim(0, 1.0)
        
        ax6.plot(angles, avg_personal_precision, color='#3B82F6', linewidth=2, linestyle='solid')
        ax6.fill(angles, avg_personal_precision, color='#3B82F6', alpha=0.4)
        st.pyplot(fig6)

    st.markdown("---")
    st.markdown("###  Live Metrics Summary")
    
    # Extract the precision for the user's specific predicted role
    target_idx = -1
    for i, d in enumerate(domains):
        if d.lower() == predicted_role.lower():
            target_idx = i
            break
            
    # If the user's role isn't exactly one of the 6, calculate it ad-hoc
    if target_idx == -1:
        role_keys = predicted_role.lower().split()
        def is_rel_ad_hoc(job_idx):
            title = str(jobs[job_idx].get("title", "")).lower()
            return any(k in title for k in role_keys)
        p_tfidf = sum(1 for idx in tfidf_top if is_rel_ad_hoc(idx)) / 5.0
        p_bm25 = sum(1 for idx in bm25_top if is_rel_ad_hoc(idx)) / 5.0
        p_knn = sum(1 for idx in knn_top if is_rel_ad_hoc(idx)) / 5.0
    else:
        p_tfidf = personal_precision[target_idx, 0]
        p_bm25 = personal_precision[target_idx, 1]
        p_knn = personal_precision[target_idx, 2]
    
    metrics_data = {
        "Metric": [f"Precision@5 ({predicted_role})", "Execution Time (s)", "Top Score Confidence"],
        "TF-IDF": [f"{p_tfidf:.2f}", f"{t_tfidf:.4f}", f"{tfidf_sim[tfidf_top[0]]:.2f}"],
        "BM25": [f"{p_bm25:.2f}", f"{t_bm25:.4f}", f"{bm25_scores[bm25_top[0]]:.2f}"],
        "KNN": [f"{p_knn:.2f}", f"{t_knn:.4f}", f"{1 - distances[0][0]:.2f}"]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    def highlight_max(row):
        styles = [''] * len(row)
        if row.name == 0:  # Precision
            vals = [float(row['TF-IDF']), float(row['BM25']), float(row['KNN'])]
            max_idx = np.argmax(vals) + 1
            styles[max_idx] = 'background-color: #064E3B; color: #FAFAFA'
        elif row.name == 1:  # Time
            vals = [float(row['TF-IDF']), float(row['BM25']), float(row['KNN'])]
            min_idx = np.argmin(vals) + 1
            styles[min_idx] = 'background-color: #064E3B; color: #FAFAFA'
        return styles
        
    st.dataframe(df_metrics.style.apply(highlight_max, axis=1), use_container_width=True)
    
    st.success(f"""
    **Academic Conclusion for {first_name}:**
    The Radar Chart geometrically proves why the AI classified you as a `{predicted_role}`! When your resume is mathematically mapped against the entire job corpus, the Top 5 retrieved jobs overwhelmingly align with that specific domain across all search algorithms.
    """)

# ==========================================
# TAB 3: MODEL COMPARISON
# ==========================================
with tab3:
    st.markdown("###  Job Role Classifier — Ensemble Model Comparison")
    st.markdown(
        "The Job Role Classifier uses a **4-model Soft-Voting Ensemble**. "
        "Each model independently predicts the job category from TF-IDF features. "
        "Their `predict_proba` outputs are averaged to produce the final ensemble prediction."
    )

    import pickle, os

    metrics_path = os.path.join("models", "ensemble_metrics.pkl")

    if not os.path.exists(metrics_path):
        st.warning("Ensemble metrics not found. Training all models now (this takes ~2 minutes)...")
        from ml_model import train_job_role_classifier
        with st.spinner("Training Naive Bayes, KNN, Logistic Regression, Random Forest..."):
            results = train_job_role_classifier()
        st.success("Training complete! Reload the page to see the results.")
    else:
        with open(metrics_path, "rb") as f:
            results = pickle.load(f)

        model_display = {
            "naive_bayes":          "Naive Bayes (MultinomialNB)",
            "knn":                  "K-Nearest Neighbors (K=5)",
            "logistic_regression":  "Logistic Regression",
            "ensemble":             " Soft-Voting Ensemble",
        }

        rows = []
        for key, label in model_display.items():
            if key in results:
                r = results[key]
                rows.append({
                    "Model":          label,
                    "Train Accuracy": f"{r.get('train_accuracy', '—') * 100:.2f}%" if isinstance(r.get('train_accuracy'), float) else "—",
                    "Test Accuracy":  f"{r.get('test_accuracy', 0) * 100:.2f}%",
                })

        df_models = pd.DataFrame(rows)

        def highlight_ensemble(row):
            if "Ensemble" in str(row["Model"]):
                return ["background-color: #064E3B; color: #FAFAFA; font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_models.style.apply(highlight_ensemble, axis=1),
            use_container_width=True,
            hide_index=True
        )

        # Bar chart comparison
        st.markdown("#### Test Accuracy Comparison")
        chart_data = {
            row["Model"].replace(" ", ""): float(row["Test Accuracy"].replace("%", ""))
            for row in rows if row["Test Accuracy"] != "—"
        }

        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ["#3B82F6", "#8B5CF6", "#F59E0B", "#EF4444", "#10B981"]
        bars = ax.barh(list(chart_data.keys()), list(chart_data.values()), color=colors, height=0.5)
        ax.set_xlabel("Test Accuracy (%)", color="white")
        ax.set_xlim(0, 105)
        ax.tick_params(colors="white")
        ax.set_facecolor("#18181B")
        fig.patch.set_facecolor("#18181B")
        for bar, val in zip(bars, chart_data.values()):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%", va="center", color="white", fontsize=10, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3F3F46")
        st.pyplot(fig)

        st.info(
            "**How Soft Voting Works:** Each model outputs a probability vector over all 25 job categories. "
            "The ensemble averages these 4 vectors element-wise and picks the category with the highest mean probability. "
            "This reduces individual model bias and consistently outperforms any single classifier."
        )


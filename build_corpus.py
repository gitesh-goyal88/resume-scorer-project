import pandas as pd
import os
import random

def process_datasets():
    print("Starting targeted data aggregation...")
    all_jobs = []
    
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    
    # 1. Process Naukri Jobs (Indian dataset)
    try:
        p2 = os.path.join(base_path, "marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv")
        df2 = pd.read_csv(p2)
        
        # Basic filters
        df2 = df2.dropna(subset=['Job Title', 'Key Skills'])
        
        for _, row in df2.iterrows():
            title = str(row.get('Job Title', 'Unknown'))
            all_jobs.append({
                "id": f"NAU_{random.randint(100000, 999999)}",
                "title": title,
                "company": "Naukri Listed Company",
                "location": str(row.get('Location', 'India')),
                "salary": str(row.get('Job Salary', 'Not Disclosed')),
                "description": f"{title} {row.get('Role Category','')} {row.get('Key Skills','')} {row.get('Functional Area','')} {row.get('Industry','')} {row.get('Job Experience Required','')}"
            })
        print(f"Loaded {len(df2)} jobs from Naukri sample")
    except Exception as e:
        print(f"Error reading Naukri dataset: {e}")

    # 2. Process LinkedIn Postings (with heavy filtering)
    try:
        p3 = os.path.join(base_path, "postings.csv")
        chunk_iter = pd.read_csv(p3, chunksize=50000)
        
        linkedin_jobs = []
        india_cities = 'Bengaluru|Mumbai|Pune|Hyderabad|Chennai|Delhi|Gurgaon|Noida|Kolkata|Ahmedabad|Remote'
        tech_roles = 'Engineer|Developer|Analyst|ML|Data|Scientist|Architect|Programmer|Software|Frontend|Backend|Full Stack|AI'
        
        for chunk in chunk_iter:
            chunk = chunk.dropna(subset=['title', 'description', 'location'])
            
            # India filter
            india_mask = chunk['location'].str.contains(india_cities, case=False, na=False)
            # Tech roles filter
            tech_mask = chunk['title'].str.contains(tech_roles, case=False, na=False)
            # Description length filter
            len_mask = chunk['description'].str.len() > 200
            
            filtered_chunk = chunk[india_mask & tech_mask & len_mask]
            
            for _, row in filtered_chunk.iterrows():
                linkedin_jobs.append({
                    "id": f"LNK_{row.get('job_id', random.randint(100000, 999999))}",
                    "title": str(row.get('title', 'Unknown')),
                    "company": str(row.get('company_name', 'Confidential')),
                    "location": str(row.get('location', 'India')),
                    "salary": str(row.get('med_salary', 'Not Disclosed')),
                    "description": str(row.get('description', ''))
                })
            
            # Stop if we have enough to sample from
            if len(linkedin_jobs) > 10000:
                break
                
        print(f"Loaded {len(linkedin_jobs)} filtered jobs from LinkedIn")
        all_jobs.extend(linkedin_jobs)
        
    except Exception as e:
        print(f"Error reading LinkedIn dataset: {e}")

    print(f"Total aggregated jobs before stratified sampling: {len(all_jobs)}")
    
    # 3. Stratified Sampling - Max 100 per title, Total 3000
    df_all = pd.DataFrame(all_jobs)
    
    # Filter descriptions > 200 length for Naukri as well
    df_all = df_all[df_all['description'].str.len() > 200]
    
    sampled_df = df_all.groupby('title').apply(
        lambda x: x.sample(min(len(x), 100))
    ).reset_index(drop=True)
    
    # Shuffle and pick exactly 3000
    sampled_df = sampled_df.sample(frac=1).reset_index(drop=True)
    final_corpus = sampled_df.head(3000).to_dict('records')
    
    # Save to clean corpus
    os.makedirs("data", exist_ok=True)
    out_path = "data/real_jobs_corpus.csv"
    pd.DataFrame(final_corpus).to_csv(out_path, index=False)
    print(f"Successfully saved {len(final_corpus)} sampled jobs to {out_path}")

if __name__ == "__main__":
    process_datasets()

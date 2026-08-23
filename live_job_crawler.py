import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import os
import time

def crawl_jobs(url="https://news.ycombinator.com/jobs"):
    """
    Crawls live software jobs from YC Hacker News Jobs to fulfill the 'Crawl' requirement.
    Uses basic BeautifulSoup HTML parsing.
    """
    print(f"Starting crawler on {url}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Crawler failed to fetch page: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Hacker News stores items in rows with class 'athing'
    job_rows = soup.find_all('tr', class_='athing')
    
    scraped_jobs = []
    for row in job_rows:
        try:
            title_elem = row.find('span', class_='titleline')
            if title_elem:
                title_text = title_elem.text.strip()
                # Simple heuristic: Split on " is hiring " or similar
                if " is hiring " in title_text:
                    parts = title_text.split(" is hiring ", 1)
                    company = parts[0].strip()
                    title = parts[1].strip()
                elif " hiring " in title_text:
                    parts = title_text.split(" hiring ", 1)
                    company = parts[0].strip()
                    title = parts[1].strip()
                else:
                    company = "YC Startup"
                    title = title_text

                description = f"Live job from Hacker News: {title_text}. Looking for strong engineering candidates."
                
                scraped_jobs.append({
                    "id": f"CRAWL_{random.randint(100000, 999999)}",
                    "title": title,
                    "company": company,
                    "location": "Remote / USA",
                    "salary": "Not Disclosed",
                    "description": description
                })
        except Exception as e:
            continue
            
    print(f"Crawler successfully scraped {len(scraped_jobs)} live jobs.")
    return scraped_jobs

def append_to_corpus():
    scraped_jobs = crawl_jobs()
    if not scraped_jobs:
        print("No jobs to append.")
        return
        
    corpus_path = "data/real_jobs_corpus.csv"
    if os.path.exists(corpus_path):
        df = pd.read_csv(corpus_path)
        new_df = pd.DataFrame(scraped_jobs)
        combined_df = pd.concat([df, new_df], ignore_index=True)
        
        # Deduplicate based on title and company
        combined_df = combined_df.drop_duplicates(subset=['title', 'company'], keep='last')
        
        combined_df.to_csv(corpus_path, index=False)
        print(f"Appended jobs to corpus. New corpus size: {len(combined_df)}")
    else:
        print(f"Corpus {corpus_path} not found.")

if __name__ == "__main__":
    append_to_corpus()

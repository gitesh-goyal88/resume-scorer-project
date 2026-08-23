import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "resumeiq.db")

def init_db():
    """Initialize the SQLite database with the candidates table."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_resumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        file_path TEXT NOT NULL,
        is_enhanced BOOLEAN DEFAULT 0,
        health_score INTEGER,
        domain TEXT,
        centroid_score REAL,
        upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT,
        phone TEXT,
        predicted_role TEXT,
        ats_score INTEGER,
        health_score INTEGER,
        strong_bullets INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company TEXT NOT NULL,
        role TEXT NOT NULL,
        status TEXT NOT NULL,
        url TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

def insert_candidate(name, email, phone, role, ats, health, bullets):
    """Insert a new processed candidate into the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO candidates (name, email, phone, predicted_role, ats_score, health_score, strong_bullets)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (name, email, phone, role, ats, health, bullets))
    
    conn.commit()
    conn.close()

def get_all_candidates():
    """Retrieve all candidates for the HR Dashboard."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM candidates ORDER BY ats_score DESC')
    rows = cursor.fetchall()
    
    # Get column names
    col_names = [description[0] for description in cursor.description]
    
    conn.close()
    
    # Convert to list of dicts
    return [dict(zip(col_names, row)) for row in rows]

def insert_application(company, role, status, url):
    """Insert a new tracked job application."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO applications (company, role, status, url)
    VALUES (?, ?, ?, ?)
    ''', (company, role, status, url))
    
    conn.commit()
    conn.close()

def get_all_applications():
    """Retrieve all tracked applications."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM applications ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    col_names = [description[0] for description in cursor.description]
    conn.close()
    
    return [dict(zip(col_names, row)) for row in rows]

def update_application_status(app_id, new_status):
    """Update the status of a tracked application."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    UPDATE applications
    SET status = ?
    WHERE id = ?
    ''', (new_status, app_id))
    
    conn.commit()
    conn.close()

# --- Auth & User Functions ---

def create_user(name, username, password_hash):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (name, username, password_hash) VALUES (?, ?, ?)', (name, username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(username):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "name": row[1], "username": row[2], "password_hash": row[3]}
    return None

def save_user_resume(user_id, file_path, is_enhanced, health_score, domain, centroid_score=0.0):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check limit of 4
    cursor.execute('SELECT id FROM user_resumes WHERE user_id = ? ORDER BY upload_date ASC', (user_id,))
    resumes = cursor.fetchall()
    
    if len(resumes) >= 4:
        # Delete oldest
        oldest_id = resumes[0][0]
        cursor.execute('DELETE FROM user_resumes WHERE id = ?', (oldest_id,))
        
    cursor.execute('''
    INSERT INTO user_resumes (user_id, file_path, is_enhanced, health_score, domain, centroid_score)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, file_path, is_enhanced, health_score, domain, centroid_score))
    
    conn.commit()
    conn.close()

def get_user_resumes(user_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM user_resumes WHERE user_id = ? ORDER BY upload_date DESC', (user_id,))
    rows = cursor.fetchall()
    col_names = [description[0] for description in cursor.description]
    conn.close()
    return [dict(zip(col_names, row)) for row in rows]

def get_leaderboard_data(domain_filter="All"):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = '''
    SELECT u.name, r.domain, r.health_score, r.centroid_score, 
           ((r.centroid_score * 0.6) + (r.health_score * 0.4)) as total_score
    FROM users u
    JOIN user_resumes r ON u.id = r.user_id
    '''
    params = []
    
    if domain_filter != "All":
        query += ' WHERE r.domain = ?'
        params.append(domain_filter)
        
    query += ' ORDER BY total_score DESC'
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    # Return highest score per user only
    leaderboard = []
    seen_users = set()
    for row in rows:
        name = row[0]
        if name not in seen_users:
            leaderboard.append({
                "Name": name,
                "Domain": row[1],
                "Health Score": row[2],
                "Centroid Score": round(row[3], 2),
                "Total Score": round(row[4], 2)
            })
            seen_users.add(name)
            
    return leaderboard

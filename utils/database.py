# /utils/database.py
import sqlite3
import pandas as pd
from datetime import datetime

class ExperimentDB:
    def __init__(self, db_path='ml_experiments.db'):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS experiments
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      model_name TEXT,
                      accuracy REAL,
                      cv_score REAL,
                      parameters TEXT,
                      dataset_shape TEXT,
                      training_time REAL)''')
        conn.commit()
        conn.close()
    
    def save_experiment(self, model_name, accuracy, cv_score, params, dataset_shape, training_time):
        """Save experiment to database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO experiments VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)",
                  (datetime.now().isoformat(), model_name, accuracy, 
                   cv_score, str(params), str(dataset_shape), training_time))
        conn.commit()
        conn.close()
    
    def load_experiments(self, limit=10):
        """Load recent experiments"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(f"SELECT * FROM experiments ORDER BY timestamp DESC LIMIT {limit}", conn)
        conn.close()
        return df
    
    def clear_experiments(self):
        """Clear all experiments"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM experiments")
        conn.commit()
        conn.close()
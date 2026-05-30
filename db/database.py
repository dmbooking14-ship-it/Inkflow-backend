"""
SQLite Local Database
For caching, ML training data, and offline operation
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

DB_PATH = Path(__file__).parent.parent / "inkflow_local.db"

class LocalDatabase:
    """Local SQLite database for caching and analysis"""
    
    def __init__(self):
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        """Create all required tables"""
        cursor = self.conn.cursor()
        
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS artists_snapshot (
                id TEXT PRIMARY KEY,
                data TEXT,
                snapshot_date TEXT,
                created_at TEXT
            );
            
            CREATE TABLE IF NOT EXISTS outreach_history (
                id TEXT PRIMARY KEY,
                action TEXT,
                platform TEXT,
                artist_name TEXT,
                result TEXT,
                message_text TEXT,
                created_at TEXT
            );
            
            CREATE TABLE IF NOT EXISTS gemini_cache (
                id TEXT PRIMARY KEY,
                question_signature TEXT,
                response TEXT,
                embedding BLOB,
                model_used TEXT,
                tokens_used INTEGER,
                created_at TEXT,
                reuse_count INTEGER DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS metrics_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                metric_name TEXT,
                metric_value REAL,
                metadata TEXT
            );
            
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_type TEXT,
                input_data TEXT,
                prediction_result TEXT,
                actual_outcome TEXT,
                confidence REAL,
                created_at TEXT,
                outcome_date TEXT
            );
            
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_type TEXT,
                report_data TEXT,
                parameters TEXT,
                created_at TEXT
            );
            
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                role TEXT,
                content TEXT,
                timestamp TEXT,
                intent TEXT,
                sentiment TEXT,
                topic TEXT
            );
            
            CREATE TABLE IF NOT EXISTS metrics_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE,
                total_artists INTEGER,
                active_artists INTEGER,
                mrr REAL,
                total_bookings INTEGER,
                bookings_7d INTEGER,
                bookings_30d INTEGER,
                dms_sent INTEGER,
                replies_received INTEGER,
                reply_rate REAL,
                health_score REAL,
                churn_rate REAL,
                conversion_rate REAL,
                created_at TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_cache_signature ON gemini_cache(question_signature);
            CREATE INDEX IF NOT EXISTS idx_metrics_date ON metrics_history(date);
            CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(prediction_type);
            CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);
        """)
        
        self.conn.commit()
    
    # ========== CACHE OPERATIONS ==========
    
    def cache_gemini_response(self, question: str, response: str, model: str, tokens: int) -> bool:
        """Cache a Gemini response"""
        try:
            signature = self._create_signature(question)
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO gemini_cache (id, question_signature, response, model_used, tokens_used, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (signature, signature, response, model, tokens, datetime.now().isoformat()))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Cache error: {e}")
            return False
    
    def get_cached_response(self, question: str) -> Optional[Dict]:
        """Get cached Gemini response"""
        signature = self._create_signature(question)
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM gemini_cache 
            WHERE question_signature = ? 
            AND created_at > datetime('now', '-7 days')
            ORDER BY created_at DESC LIMIT 1
        """, (signature,))
        row = cursor.fetchone()
        if row:
            # Update reuse count
            cursor.execute("UPDATE gemini_cache SET reuse_count = reuse_count + 1 WHERE id = ?", (row["id"],))
            self.conn.commit()
            return dict(row)
        return None
    
    def search_similar_cache(self, question: str, threshold: float = 0.7) -> Optional[Dict]:
        """Search for semantically similar cached responses"""
        signature = self._create_signature(question)
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM gemini_cache 
            WHERE created_at > datetime('now', '-7 days')
            ORDER BY created_at DESC LIMIT 50
        """)
        rows = cursor.fetchall()
        
        best_match = None
        best_score = 0
        
        for row in rows:
            score = self._similarity(signature, row["question_signature"])
            if score > best_score and score > threshold:
                best_score = score
                best_match = dict(row)
        
        return best_match
    
    # ========== METRICS OPERATIONS ==========
    
    def save_daily_metrics(self, metrics: Dict) -> bool:
        """Save daily metrics snapshot"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO metrics_snapshots 
                (date, total_artists, active_artists, mrr, total_bookings, bookings_7d, bookings_30d,
                 dms_sent, replies_received, reply_rate, health_score, churn_rate, conversion_rate, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.get("date", datetime.now().strftime("%Y-%m-%d")),
                metrics.get("totalArtists", 0),
                metrics.get("activeArtists", 0),
                metrics.get("mrr", 0),
                metrics.get("totalBookings", 0),
                metrics.get("bookings7d", 0),
                metrics.get("bookings30d", 0),
                metrics.get("dmsSent", 0),
                metrics.get("repliesReceived", 0),
                metrics.get("replyRate", 0),
                metrics.get("healthScore", 0),
                metrics.get("churnRate", 0),
                metrics.get("conversionRate", 0),
                datetime.now().isoformat()
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Metrics save error: {e}")
            return False
    
    def get_metrics_history(self, days: int = 30) -> List[Dict]:
        """Get metrics history for trend analysis"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM metrics_snapshots 
            WHERE date > date('now', ?) 
            ORDER BY date ASC
        """, (f"-{days} days",))
        return [dict(row) for row in cursor.fetchall()]
    
    # ========== PREDICTION OPERATIONS ==========
    
    def save_prediction(self, pred_type: str, input_data: Dict, result: Dict, confidence: float) -> bool:
        """Save a prediction for later validation"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (prediction_type, input_data, prediction_result, confidence, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (pred_type, json.dumps(input_data), json.dumps(result), confidence, datetime.now().isoformat()))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Prediction save error: {e}")
            return False
    
    def get_predictions_for_training(self, pred_type: str = None) -> List[Dict]:
        """Get predictions with actual outcomes for ML training"""
        cursor = self.conn.cursor()
        if pred_type:
            cursor.execute("""
                SELECT * FROM predictions 
                WHERE prediction_type = ? AND actual_outcome IS NOT NULL
                ORDER BY created_at DESC LIMIT 500
            """, (pred_type,))
        else:
            cursor.execute("""
                SELECT * FROM predictions 
                WHERE actual_outcome IS NOT NULL
                ORDER BY created_at DESC LIMIT 500
            """)
        return [dict(row) for row in cursor.fetchall()]
    
    def update_prediction_outcome(self, prediction_id: int, actual_outcome: str) -> bool:
        """Update a prediction with its actual outcome"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE predictions 
                SET actual_outcome = ?, outcome_date = ?
                WHERE id = ?
            """, (actual_outcome, datetime.now().isoformat(), prediction_id))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Outcome update error: {e}")
            return False
    
    # ========== CONVERSATION OPERATIONS ==========
    
    def save_conversation(self, msg_id: str, role: str, content: str, intent: str = None, sentiment: str = None) -> bool:
        """Save conversation message"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO conversations (id, role, content, timestamp, intent, sentiment)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (msg_id, role, content, datetime.now().isoformat(), intent, sentiment))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Conversation save error: {e}")
            return False
    
    # ========== REPORT OPERATIONS ==========
    
    def save_report(self, report_type: str, report_data: Dict, parameters: Dict = None) -> int:
        """Save generated report"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO reports (report_type, report_data, parameters, created_at)
            VALUES (?, ?, ?, ?)
        """, (report_type, json.dumps(report_data), json.dumps(parameters or {}), datetime.now().isoformat()))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_recent_reports(self, report_type: str = None, limit: int = 10) -> List[Dict]:
        """Get recent reports"""
        cursor = self.conn.cursor()
        if report_type:
            cursor.execute("""
                SELECT * FROM reports WHERE report_type = ? 
                ORDER BY created_at DESC LIMIT ?
            """, (report_type, limit))
        else:
            cursor.execute("""
                SELECT * FROM reports ORDER BY created_at DESC LIMIT ?
            """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    # ========== UTILITY ==========
    
    def _create_signature(self, text: str) -> str:
        """Create a normalized signature for caching"""
        return text.lower().strip()[:100]
    
    def _similarity(self, a: str, b: str) -> float:
        """Simple string similarity score"""
        a = a.lower()
        b = b.lower()
        if len(a) == 0 or len(b) == 0:
            return 0.0
        longer = a if len(a) > len(b) else b
        shorter = b if len(a) > len(b) else a
        matches = sum(1 for c in shorter if c in longer)
        return matches / len(longer)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

# Singleton instance
local_db = LocalDatabase()

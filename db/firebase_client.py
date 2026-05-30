"""
Firebase Firestore Client
Handles all database operations with connection pooling and caching
"""

import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
from functools import lru_cache
import pandas as pd

class FirebaseClient:
    """Singleton Firebase client with caching and connection management"""
    
    _instance = None
    _db = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._db is None:
            self._initialize()
    
    def _initialize(self):
        """Initialize Firebase connection"""
        try:
            cred = credentials.Certificate("firebase_key.json")
            firebase_admin.initialize_app(cred)
            self._db = firestore.client()
            print("✅ Firebase connected")
        except Exception as e:
            print(f"❌ Firebase init error: {e}")
            raise
    
    @property
    def db(self):
        return self._db
    
    # ========== COLLECTION READERS ==========
    
    def get_all_artists(self) -> List[Dict]:
        """Fetch all artists with caching"""
        try:
            docs = self._db.collection("artists").get()
            artists = []
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id
                artists.append(data)
            return artists
        except Exception as e:
            print(f"Error fetching artists: {e}")
            return []
    
    def get_all_bookings(self) -> List[Dict]:
        """Fetch all bookings"""
        try:
            docs = self._db.collection("bookings").get()
            bookings = []
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id
                bookings.append(data)
            return bookings
        except Exception as e:
            print(f"Error fetching bookings: {e}")
            return []
    
    def get_outreach_logs(self, limit: int = 500) -> List[Dict]:
        """Fetch outreach logs with limit"""
        try:
            docs = self._db.collection("outreach_logs")\
                .order_by("createdAt", direction=firestore.Query.DESCENDING)\
                .limit(limit)\
                .get()
            logs = []
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id
                logs.append(data)
            return logs
        except Exception as e:
            print(f"Error fetching outreach: {e}")
            return []
    
    def get_pending_payments(self) -> List[Dict]:
        """Fetch pending payments"""
        try:
            docs = self._db.collection("pending_payments")\
                .order_by("createdAt", direction=firestore.Query.DESCENDING)\
                .get()
            payments = []
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id
                payments.append(data)
            return payments
        except Exception as e:
            print(f"Error fetching payments: {e}")
            return []
    
    def get_whatsapp_stats(self) -> List[Dict]:
        """Fetch WhatsApp stats"""
        try:
            docs = self._db.collection("whatsapp_stats").get()
            stats = []
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id
                stats.append(data)
            return stats
        except Exception as e:
            print(f"Error fetching WhatsApp: {e}")
            return []
    
    def get_side_hustle(self) -> List[Dict]:
        """Fetch side hustle projects"""
        try:
            docs = self._db.collection("side_hustle").get()
            projects = []
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id
                projects.append(data)
            return projects
        except Exception as e:
            print(f"Error fetching side hustle: {e}")
            return []
    
    def get_honeypot_logs(self, limit: int = 100) -> List[Dict]:
        """Fetch security logs"""
        try:
            docs = self._db.collection("honeypot_logs")\
                .order_by("timestamp", direction=firestore.Query.DESCENDING)\
                .limit(limit)\
                .get()
            logs = []
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id
                logs.append(data)
            return logs
        except Exception as e:
            print(f"Error fetching honeypot: {e}")
            return []
    
    def get_competition_winners(self) -> List[Dict]:
        """Fetch competition winners"""
        try:
            docs = self._db.collection("competition_winners").get()
            winners = []
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id
                winners.append(data)
            return winners
        except Exception as e:
            print(f"Error fetching winners: {e}")
            return []
    
    def get_daily_snapshots(self, limit: int = 90) -> List[Dict]:
        """Fetch daily snapshots"""
        try:
            docs = self._db.collection("daily_snapshots")\
                .order_by("createdAt", direction=firestore.Query.DESCENDING)\
                .limit(limit)\
                .get()
            snapshots = []
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id
                snapshots.append(data)
            return snapshots
        except Exception as e:
            print(f"Error fetching snapshots: {e}")
            return []
    
    # ========== DATA WRITERS ==========
    
    def save_prediction(self, prediction_data: Dict) -> bool:
        """Save ML prediction to Firestore"""
        try:
            self._db.collection("ml_predictions").add({
                **prediction_data,
                "createdAt": firestore.SERVER_TIMESTAMP
            })
            return True
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return False
    
    def save_report(self, report_data: Dict) -> bool:
        """Save generated report"""
        try:
            self._db.collection("generated_reports").add({
                **report_data,
                "createdAt": firestore.SERVER_TIMESTAMP
            })
            return True
        except Exception as e:
            print(f"Error saving report: {e}")
            return False
    
    def take_snapshot(self) -> Dict:
        """Take a complete dashboard snapshot"""
        artists = self.get_all_artists()
        bookings = self.get_all_bookings()
        
        total = len(artists)
        active = len([a for a in artists if a.get("status") == "active"])
        trial = len([a for a in artists if a.get("status") == "trial"])
        expired = len([a for a in artists if a.get("status") == "expired"])
        
        standard = len([a for a in artists if a.get("tier", "standard") == "standard" and a.get("status") == "active"])
        pro = len([a for a in artists if a.get("tier") == "pro" and a.get("status") == "active"])
        premium = len([a for a in artists if a.get("tier") == "premium" and a.get("status") == "active"])
        
        mrr = (standard * 19) + (pro * 39) + (premium * 59)
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "totalArtists": total,
            "activeArtists": active,
            "trialArtists": trial,
            "expiredArtists": expired,
            "standardActive": standard,
            "proActive": pro,
            "premiumActive": premium,
            "mrr": mrr,
            "totalBookings": len(bookings),
            "approvedBookings": len([b for b in bookings if b.get("status") == "approved"]),
            "pendingBookings": len([b for b in bookings if not b.get("status") or b.get("status") == "pending"]),
            "declinedBookings": len([b for b in bookings if b.get("status") == "declined"])
        }
        
        # Save to Firestore
        try:
            self._db.collection("daily_snapshots").add({
                **snapshot,
                "createdAt": firestore.SERVER_TIMESTAMP
            })
        except:
            pass
        
        return snapshot

# Singleton instance
firebase_client = FirebaseClient()

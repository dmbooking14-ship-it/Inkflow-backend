"""
InkFlow Machine Learning Engine
ML features temporarily stubbed — will be fully enabled when dependencies are added
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

class MLEngine:
    """Complete machine learning pipeline for InkFlow (stubbed for now)"""
    
    def __init__(self):
        self.models = {}
        self.last_trained = None
        print("⚠️ ML Engine running in stub mode — install sklearn/xgboost for full features")
    
    def extract_artist_features(self, artist: Dict) -> List[float]:
        """Extract numerical features from artist data (basic version)"""
        features = []
        followers = artist.get('followers', 0) or 0
        features.append(min(followers / 10000, 1.0))
        booking = (artist.get('booking_method') or '').lower()
        features.append(1.0 if 'dm' in booking else 0.0)
        features.append(1.0 if 'whatsapp' in booking else 0.0)
        status = artist.get('status', 'trial')
        features.append(1.0 if status == 'active' else 0.0)
        features.append(1.0 if status == 'trial' else 0.0)
        tier = artist.get('tier', 'standard')
        features.append(1.0 if tier == 'standard' else 0.0)
        features.append(1.0 if tier == 'pro' else 0.0)
        features.append(1.0 if tier == 'premium' else 0.0)
        features.append(1.0 if artist.get('referralCount', 0) > 0 else 0.0)
        return features
    
    def score_lead(self, artist: Dict) -> Dict:
        """Score a single lead (basic heuristic version)"""
        score = 0
        followers = artist.get('followers', 0) or 0
        if followers > 5000: score += 3
        elif followers > 1000: score += 2
        elif followers > 500: score += 1
        booking = (artist.get('booking_method') or '').lower()
        if 'dm' in booking: score += 3
        if 'whatsapp' in booking: score += 2
        status = artist.get('status', '')
        if status == 'active': score += 1
        max_score = 7
        normalized = min(score / max_score, 1.0)
        quality = "Excellent" if normalized > 0.7 else "Good" if normalized > 0.5 else "Fair" if normalized > 0.3 else "Low"
        return {"score": round(normalized, 3), "quality": quality, "confidence": 0.5, "status": "heuristic"}
    
    def predict_churn_risk(self, artist: Dict) -> Dict:
        """Predict churn risk (basic heuristic version)"""
        risk = 0
        status = artist.get('status', '')
        if status == 'expired': risk = 1.0
        elif status == 'trial': risk = 0.4
        elif status == 'active': risk = 0.1
        level = "High Risk" if risk > 0.7 else "Medium Risk" if risk > 0.4 else "Low Risk"
        return {"risk_score": round(risk, 3), "risk_level": level, "confidence": 0.5, "status": "heuristic"}
    
    def grade_message(self, message_text: str, platform: str = "instagram") -> Dict:
        """Grade a draft message (basic heuristic version)"""
        score = 5
        lower = message_text.lower()
        if len(message_text) < 300: score += 1
        if '?' in message_text: score += 1
        if 'saw' in lower or 'noticed' in lower: score += 1
        if 'no pressure' in lower: score += 1
        if 'free' in lower: score += 1
        if 'http' in lower: score -= 1
        grade = "A" if score >= 8 else "B" if score >= 6 else "C" if score >= 4 else "D"
        suggestions = []
        if len(message_text) > 300: suggestions.append("Shorten message")
        if '?' not in message_text: suggestions.append("Add a question")
        if 'http' in lower: suggestions.append("Remove links")
        return {"score": min(score, 10), "grade": grade, "suggestions": suggestions, "confidence": 0.5}
    
    def detect_anomalies(self, current_metrics: Dict, historical_metrics: List[Dict]) -> List[Dict]:
        """Detect anomalies (basic version)"""
        anomalies = []
        if current_metrics.get('bookings7d', 0) == 0 and current_metrics.get('totalBookings', 0) > 0:
            anomalies.append({"metric": "bookings_48h", "current_value": 0, "severity": "high", "message": "No recent bookings"})
        return anomalies
    
    def train_lead_scorer(self, artists, outreach):
        return {"status": "stub_mode", "message": "Install sklearn for ML training"}
    
    def train_churn_predictor(self, artists, bookings):
        return {"status": "stub_mode", "message": "Install sklearn for ML training"}
    
    def train_message_grader(self, outreach):
        return {"status": "stub_mode", "message": "Install sklearn for ML training"}
    
    def retrain_all(self, artists, bookings, outreach):
        return {"status": "stub_mode", "total_artists": len(artists)}


ml_engine = MLEngine()

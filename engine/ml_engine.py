"""
InkFlow Machine Learning Engine
Lead scoring, churn prediction, message grading, anomaly detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

class MLEngine:
    """Complete machine learning pipeline for InkFlow"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.feature_importance = {}
        self.last_trained = None
        self._load_models()
    
    def _load_models(self):
        """Load saved models if they exist"""
        try:
            for model_name in ['lead_scorer', 'churn_predictor', 'message_grader', 'conversion_predictor']:
                model_path = MODEL_DIR / f"{model_name}.pkl"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
            print(f"✅ Loaded {len(self.models)} ML models")
        except Exception as e:
            print(f"⚠️ Could not load models: {e}")
    
    def _save_model(self, model_name: str, model):
        """Save model to disk"""
        try:
            model_path = MODEL_DIR / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"⚠️ Could not save model: {e}")
    
    # ========== FEATURE ENGINEERING ==========
    
    def extract_artist_features(self, artist: Dict) -> np.ndarray:
        """Extract numerical features from artist data for ML"""
        features = []
        
        # Follower count (normalized)
        followers = artist.get('followers', 0) or 0
        features.append(min(followers / 10000, 1.0))
        
        # Booking method encoding
        booking = (artist.get('booking_method') or '').lower()
        features.append(1.0 if 'dm' in booking else 0.0)
        features.append(1.0 if 'whatsapp' in booking else 0.0)
        features.append(1.0 if 'link' in booking else 0.0)
        
        # Status encoding
        status = artist.get('status', 'trial')
        features.append(1.0 if status == 'active' else 0.0)
        features.append(1.0 if status == 'trial' else 0.0)
        features.append(1.0 if status == 'expired' else 0.0)
        
        # Tier encoding
        tier = artist.get('tier', 'standard')
        features.append(1.0 if tier == 'standard' else 0.0)
        features.append(1.0 if tier == 'pro' else 0.0)
        features.append(1.0 if tier == 'premium' else 0.0)
        
        # Activity indicators
        features.append(1.0 if artist.get('bookingLink') else 0.0)
        features.append(1.0 if artist.get('referralCount', 0) > 0 else 0.0)
        features.append(min((artist.get('referralCount', 0) or 0) / 10, 1.0))
        
        # Trial timing
        if artist.get('trialEndsAt'):
            try:
                trial_end = artist['trialEndsAt']
                if hasattr(trial_end, 'timestamp'):
                    days_left = (trial_end.timestamp() - datetime.now().timestamp()) / 86400
                else:
                    days_left = 30
                features.append(max(min(days_left / 30, 1.0), 0.0))
            except:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # Age of account
        if artist.get('createdAt'):
            try:
                created = artist['createdAt']
                if hasattr(created, 'timestamp'):
                    age_days = (datetime.now().timestamp() - created.timestamp()) / 86400
                else:
                    age_days = 0
                features.append(min(age_days / 90, 1.0))
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def extract_booking_features(self, artist: Dict, bookings: List[Dict], outreach: List[Dict]) -> np.ndarray:
        """Extract features combining artist data with booking and outreach history"""
        artist_features = self.extract_artist_features(artist)
        
        # Booking features
        artist_bookings = [b for b in bookings if b.get('artistEmail') == artist.get('email')]
        total_bookings = len(artist_bookings)
        approved = len([b for b in artist_bookings if b.get('status') == 'approved'])
        pending = len([b for b in artist_bookings if not b.get('status') or b.get('status') == 'pending'])
        
        booking_features = [
            min(total_bookings / 50, 1.0),
            approved / max(total_bookings, 1),
            pending / max(total_bookings, 1),
            1.0 if total_bookings > 0 else 0.0
        ]
        
        # Outreach features
        artist_outreach = [o for o in outreach if (o.get('artistName') or '').lower() == (artist.get('username') or '').lower()]
        dms_sent = len([o for o in artist_outreach if o.get('action') == 'dm_sent'])
        replies = len([o for o in artist_outreach if o.get('action') == 'reply_received'])
        
        outreach_features = [
            min(dms_sent / 20, 1.0),
            replies / max(dms_sent, 1),
            1.0 if replies > 0 else 0.0
        ]
        
        return np.concatenate([artist_features, booking_features, outreach_features])
    
    # ========== LEAD SCORING MODEL ==========
    
    def train_lead_scorer(self, artists: List[Dict], outreach: List[Dict]) -> Dict:
        """Train lead scoring model to predict which artists will respond"""
        # Build training data
        X = []
        y = []
        
        for artist in artists:
            features = self.extract_artist_features(artist)
            
            # Label: did they respond?
            artist_name = (artist.get('username') or '').lower()
            replied = any(
                o.get('action') == 'reply_received' and 
                (o.get('artistName') or '').lower() == artist_name
                for o in outreach
            )
            
            X.append(features)
            y.append(1 if replied else 0)
        
        if len(X) < 10:
            return {"status": "insufficient_data", "samples": len(X)}
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Save model
        self.models['lead_scorer'] = model
        self._save_model('lead_scorer', model)
        self.last_trained = datetime.now().isoformat()
        
        # Feature importance
        importance = model.feature_importances_.tolist()
        
        return {
            "status": "trained",
            "samples": len(X),
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "feature_importance": importance
        }
    
    def score_lead(self, artist: Dict) -> Dict:
        """Score a single lead"""
        if 'lead_scorer' not in self.models:
            return {"score": 0.5, "confidence": 0, "status": "no_model"}
        
        features = self.extract_artist_features(artist).reshape(1, -1)
        model = self.models['lead_scorer']
        
        probability = model.predict_proba(features)[0]
        score = probability[1] if len(probability) > 1 else 0.5
        
        # Determine quality tier
        if score > 0.7:
            quality = "Excellent"
        elif score > 0.5:
            quality = "Good"
        elif score > 0.3:
            quality = "Fair"
        else:
            quality = "Low"
        
        return {
            "score": round(float(score), 3),
            "quality": quality,
            "confidence": round(max(probability), 3)
        }
    
    # ========== CHURN PREDICTION MODEL ==========
    
    def train_churn_predictor(self, artists: List[Dict], bookings: List[Dict]) -> Dict:
        """Train model to predict which artists will churn"""
        X = []
        y = []
        
        for artist in artists:
            features = self.extract_artist_features(artist)
            
            # Label: have they churned?
            churned = artist.get('status') == 'expired'
            
            # Also label inactive artists with old trial dates as at-risk
            if artist.get('status') == 'trial' and artist.get('trialEndsAt'):
                try:
                    trial_end = artist['trialEndsAt']
                    if hasattr(trial_end, 'timestamp'):
                        days_left = (trial_end.timestamp() - datetime.now().timestamp()) / 86400
                        if days_left < 0:
                            churned = True
                except:
                    pass
            
            X.append(features)
            y.append(1 if churned else 0)
        
        if len(X) < 10:
            return {"status": "insufficient_data", "samples": len(X)}
        
        X = np.array(X)
        y = np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        self.models['churn_predictor'] = model
        self._save_model('churn_predictor', model)
        
        return {
            "status": "trained",
            "samples": len(X),
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 3)
        }
    
    def predict_churn_risk(self, artist: Dict) -> Dict:
        """Predict churn risk for an artist"""
        if 'churn_predictor' not in self.models:
            return {"risk_score": 0.5, "risk_level": "Unknown", "status": "no_model"}
        
        features = self.extract_artist_features(artist).reshape(1, -1)
        model = self.models['churn_predictor']
        
        probability = model.predict_proba(features)[0]
        risk = probability[1] if len(probability) > 1 else 0.5
        
        if risk > 0.7:
            level = "High Risk"
        elif risk > 0.4:
            level = "Medium Risk"
        else:
            level = "Low Risk"
        
        return {
            "risk_score": round(float(risk), 3),
            "risk_level": level,
            "confidence": round(max(probability), 3)
        }
    
    # ========== MESSAGE GRADER ==========
    
    def train_message_grader(self, outreach: List[Dict]) -> Dict:
        """Train model to grade outreach messages based on past success"""
        X = []
        y = []
        
        for entry in outreach:
            features = self._extract_message_features(entry)
            success = 1 if entry.get('action') == 'reply_received' or entry.get('result') == 'replied' else 0
            
            X.append(features)
            y.append(success)
        
        if len(X) < 10:
            return {"status": "insufficient_data", "samples": len(X)}
        
        X = np.array(X)
        y = np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        self.models['message_grader'] = model
        self._save_model('message_grader', model)
        
        return {
            "status": "trained",
            "samples": len(X),
            "accuracy": round(accuracy_score(y_test, y_pred), 3)
        }
    
    def grade_message(self, message_text: str, platform: str = "instagram") -> Dict:
        """Grade a draft message for effectiveness"""
        if 'message_grader' not in self.models:
            return {"score": 5, "grade": "N/A", "status": "no_model"}
        
        features = self._extract_message_text_features(message_text, platform)
        features = np.array(features).reshape(1, -1)
        
        model = self.models['message_grader']
        try:
            probability = model.predict_proba(features)[0]
            score = probability[1] if len(probability) > 1 else 0.5
        except:
            score = 0.5
        
        grade = "A" if score > 0.8 else "B" if score > 0.6 else "C" if score > 0.4 else "D" if score > 0.2 else "F"
        
        suggestions = []
        if len(message_text) > 300:
            suggestions.append("Shorten message — keep under 250 characters")
        if '?' not in message_text:
            suggestions.append("Add a question to encourage response")
        if 'http' in message_text.lower():
            suggestions.append("Remove links — they can trigger spam filters")
        if 'free' not in message_text.lower():
            suggestions.append("Mention the free tier to reduce friction")
        
        return {
            "score": round(float(score) * 10, 1),
            "grade": grade,
            "suggestions": suggestions,
            "confidence": round(max(probability) if 'probability' in dir() else 0.5, 3)
        }
    
    def _extract_message_features(self, entry: Dict) -> List[float]:
        """Extract features from an outreach entry"""
        text = entry.get('messageText', entry.get('message', ''))
        return self._extract_message_text_features(
            text,
            entry.get('platform', 'instagram')
        )
    
    def _extract_message_text_features(self, text: str, platform: str) -> List[float]:
        """Extract numerical features from message text"""
        lower = text.lower()
        return [
            float(len(text)),
            float(min(len(text) / 500, 1.0)),
            float(1.0 if '?' in text else 0.0),
            float(1.0 if 'saw' in lower or 'noticed' in lower or 'posted' in lower else 0.0),
            float(1.0 if 'quick question' in lower else 0.0),
            float(1.0 if 'no pressure' in lower else 0.0),
            float(1.0 if 'free' in lower else 0.0),
            float(1.0 if 'instagram' in platform.lower() else 0.0),
            float(1.0 if 'whatsapp' in platform.lower() else 0.0),
            float(1.0 if 'http' in lower else 0.0),
            float(1.0 if '!' in text else 0.0),
            float(text.count('?')),
            float(len(text.split())) / 50,
            float(1.0 if 'hey' in lower or 'hi' in lower or 'hello' in lower else 0.0),
            float(1.0 if 'book' in lower or 'booking' in lower or 'schedule' in lower else 0.0)
        ]
    
    # ========== ANOMALY DETECTION ==========
    
    def detect_anomalies(self, current_metrics: Dict, historical_metrics: List[Dict]) -> List[Dict]:
        """Detect anomalies by comparing current metrics to historical patterns"""
        anomalies = []
        
        if not historical_metrics or len(historical_metrics) < 3:
            return anomalies
        
        # Calculate baseline (mean and std of historical data)
        for metric in ['totalBookings', 'bookings7d', 'dmsSent', 'repliesReceived']:
            values = [m.get(metric, 0) for m in historical_metrics if m.get(metric) is not None]
            if not values:
                continue
            
            mean = np.mean(values)
            std = np.std(values)
            current = current_metrics.get(metric, 0)
            
            if std > 0:
                z_score = (current - mean) / std
                
                if abs(z_score) > 2.5:
                    anomalies.append({
                        "metric": metric,
                        "current_value": current,
                        "expected_range": f"{mean - 2*std:.1f} - {mean + 2*std:.1f}",
                        "z_score": round(z_score, 2),
                        "severity": "high" if abs(z_score) > 3 else "medium",
                        "direction": "spike" if z_score > 0 else "drop"
                    })
        
        # Check for specific conditions
        if current_metrics.get('bookings7d', 0) == 0 and current_metrics.get('totalBookings', 0) > 0:
            anomalies.append({
                "metric": "bookings_48h",
                "current_value": 0,
                "severity": "high",
                "message": "No bookings in the last 48 hours despite having active artists"
            })
        
        if current_metrics.get('emailPercent', 0) > 80:
            anomalies.append({
                "metric": "emailjs_usage",
                "current_value": current_metrics.get('emailPercent', 0),
                "severity": "high",
                "message": f"EmailJS at {current_metrics.get('emailPercent', 0)}% — approaching monthly limit"
            })
        
        return anomalies
    
    # ========== MODEL RETRAINING ==========
    
    def retrain_all(self, artists: List[Dict], bookings: List[Dict], outreach: List[Dict]) -> Dict:
        """Retrain all models"""
        results = {}
        
        lead_result = self.train_lead_scorer(artists, outreach)
        results['lead_scorer'] = lead_result
        
        churn_result = self.train_churn_predictor(artists, bookings)
        results['churn_predictor'] = churn_result
        
        message_result = self.train_message_grader(outreach)
        results['message_grader'] = message_result
        
        results['trained_at'] = datetime.now().isoformat()
        results['total_artists'] = len(artists)
        results['total_bookings'] = len(bookings)
        results['total_outreach'] = len(outreach)
        
        return results


# Singleton instance
ml_engine = MLEngine()

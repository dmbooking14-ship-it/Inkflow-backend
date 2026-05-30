"""
InkFlow Backend Configuration
All settings, API keys, and constants
"""

import os
from pathlib import Path

# ========== PROJECT PATHS ==========
BASE_DIR = Path(__file__).resolve().parent
FIREBASE_KEY_PATH = BASE_DIR / "firebase_key.json"
DATABASE_PATH = BASE_DIR / "inkflow_local.db"

# ========== FIREBASE CONFIG ==========
FIREBASE_PROJECT_ID = "inkflow-534f6"
FIREBASE_COLLECTIONS = [
    "artists",
    "bookings",
    "outreach_logs",
    "pending_payments",
    "whatsapp_stats",
    "side_hustle",
    "honeypot_logs",
    "competition_winners",
    "daily_snapshots",
    "admin_messages"
]

# ========== GEMINI CONFIG ==========
GEMINI_API_KEYS = [
    "AIzaSyBnICPKAzFJ4zc4mROB5k5rf0QdwUgM9Nc",
    "AIzaSyD-6AV_aHljhGwcyk_5jg1EEU1XxCYSUPU",
    "AIzaSyDZWlTSQbFRV_6C5JemPaI_H_b1VLhJQaY",
    "AIzaSyDYwR4Jc_Pdrt4Qf5Uah3LdKIDKe2mH5hI",
    "AIzaSyDxbB1CYSg-97iyNfi2Gx_hCnvhvNqHRPk"
]
GEMINI_MODEL_FLASH = "gemini-2.5-flash"
GEMINI_MODEL_PRO = "gemini-2.5-pro"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

# ========== PRICING ==========
PRICING_TIERS = {
    "standard": 19,
    "pro": 39,
    "premium": 59
}

# ========== ML MODEL SETTINGS ==========
ML_MIN_TRAINING_SAMPLES = 20
ML_RETRAIN_INTERVAL_HOURS = 24
ML_CONFIDENCE_THRESHOLD = 0.65

# ========== SCHEDULER SETTINGS ==========
SCHEDULE_DAILY_BRIEFING = "08:00"
SCHEDULE_WEEKLY_REPORT = "sun@18:00"
SCHEDULE_MONTHLY_REVIEW = "1st@09:00"
SCHEDULE_HEALTH_CHECK = "*/6"  # Every 6 hours
SCHEDULE_COMPETITOR_CHECK = "mon@10:00"

# ========== BUSINESS CONTEXT (Embedded Knowledge) ==========
BUSINESS_CONTEXT = {
    "company": "InkFlow",
    "founder": "Abdulkareem (17, Nigeria)",
    "product": "SaaS booking platform for tattoo artists",
    "current_stage": "Stage 0: Validation",
    "pricing": {"standard": 19, "pro": 39, "premium": 59},
    "system6_trigger": "10 free users OR $100 MRR",
    "competitors": ["SlidInk", "INKFLO", "InkFlow Studio", "Keep the Fees"],
    "vision": "Become the standard booking infrastructure for independent tattoo artists worldwide.",
    "exit_target": "$15-30M valuation within 5 years"
}

# ========== ANOMALY THRESHOLDS ==========
ANOMALY_CONFIG = {
    "no_bookings_48h": True,
    "emailjs_warning_percent": 80,
    "no_signups_7d": True,
    "churn_spike_percent": 10,
    "outreach_drop_percent": 50
}

# ========== API SECURITY ==========
API_RATE_LIMIT = 100  # requests per minute
API_TOKEN_EXPIRY_HOURS = 24
CORS_ORIGINS = ["*"]  # Restrict in production
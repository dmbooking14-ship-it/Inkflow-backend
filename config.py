import os

# ========== FIREBASE CONFIG ==========
FIREBASE_PROJECT_ID = "inkflow-534f6"
FIREBASE_KEY_PATH = os.environ.get("FIREBASE_KEY_PATH", "firebase_key.json")

# ========== FIREBASE COLLECTIONS ==========
COLLECTION_ARTISTS = "artists"
COLLECTION_BOOKINGS = "bookings"
COLLECTION_OUTREACH_LOGS = "outreach_logs"
COLLECTION_PENDING_PAYMENTS = "pending_payments"
COLLECTION_WHATSAPP_STATS = "whatsapp_stats"
COLLECTION_SIDE_HUSTLE = "side_hustle"
COLLECTION_HONEYPOT_LOGS = "honeypot_logs"
COLLECTION_COMPETITION_WINNERS = "competition_winners"
COLLECTION_DAILY_SNAPSHOTS = "daily_snapshots"

# ========== GEMINI API KEYS ==========
GEMINI_API_KEYS = [
    os.environ.get("GEMINI_KEY_1", ""),
    os.environ.get("GEMINI_KEY_2", ""),
    os.environ.get("GEMINI_KEY_3", ""),
    os.environ.get("GEMINI_KEY_4", ""),
    os.environ.get("GEMINI_KEY_5", ""),
]
GEMINI_API_KEYS = [k for k in GEMINI_API_KEYS if k]  # Remove empty strings

# ========== DEEPSEEK API KEYS ==========
DEEPSEEK_API_KEYS = [
    os.environ.get("DEEPSEEK_KEY_1", ""),
    os.environ.get("DEEPSEEK_KEY_2", ""),
    os.environ.get("DEEPSEEK_KEY_3", ""),
    os.environ.get("DEEPSEEK_KEY_4", ""),
    os.environ.get("DEEPSEEK_KEY_5", ""),
]
DEEPSEEK_API_KEYS = [k for k in DEEPSEEK_API_KEYS if k]

# ========== OPENROUTER API KEY ==========
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_KEY", "")

# ========== GEMINI MODELS ==========
GEMINI_MODEL_FLASH = "gemini-2.5-flash"
GEMINI_MODEL_PRO = "gemini-2.5-pro"

# ========== DEEPSEEK MODEL ==========
DEEPSEEK_MODEL = "deepseek-chat"

# ========== API ENDPOINTS ==========
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# ========== PRICING TIERS ==========
PRICING = {
    "standard": 19,
    "pro": 39,
    "premium": 59,
}

# ========== SCHEDULER TIMES (24h UTC) ==========
SCHEDULE_DAILY_BRIEFING_HOUR = 8
SCHEDULE_DAILY_BRIEFING_MINUTE = 0
SCHEDULE_WEEKLY_REPORT_DAY = "sun"
SCHEDULE_WEEKLY_REPORT_HOUR = 18
SCHEDULE_WEEKLY_REPORT_MINUTE = 0
SCHEDULE_MONTHLY_REVIEW_DAY = 1
SCHEDULE_MONTHLY_REVIEW_HOUR = 9
SCHEDULE_MONTHLY_REVIEW_MINUTE = 0
SCHEDULE_HEALTH_CHECK_INTERVAL_HOURS = 6
SCHEDULE_COMPETITOR_CHECK_DAY = "mon"
SCHEDULE_COMPETITOR_CHECK_HOUR = 10
SCHEDULE_COMPETITOR_CHECK_MINUTE = 0
SCHEDULE_DAILY_SNAPSHOT_HOUR = 0
SCHEDULE_DAILY_SNAPSHOT_MINUTE = 0

# ========== BUSINESS CONTEXT (for AI prompts) ==========
BUSINESS_CONTEXT = {
    "company_name": "InkFlow",
    "founder": "Abdulkareem (17, Nigeria)",
    "product": "SaaS booking platform for tattoo artists",
    "current_version": "System 5 (free, live)",
    "next_version": "System 6 (complete, ready to launch)",
    "stage": "Stage 0: Validation",
    "paying_customers": 0,
    "mrr": 0,
    "primary_challenge": "Customer acquisition (no budget, banned from multiple platforms)",
    "working_channels": ["Instagram DMs", "WhatsApp"],
    "blocked_channels": ["Reddit (banned)", "Email (paused)"],
    "system6_trigger": "10 free users OR $100 MRR",
    "pricing_tiers": PRICING,
    "competitors": {
        "slidink": {
            "pricing": "$29-225/month",
            "strengths": ["IG DM integration", "polished UI", "referral program"],
            "weaknesses": ["Higher price", "no approval-first", "no bilingual"],
        },
        "inkflo": {
            "pricing": "$9-59/month",
            "strengths": ["Low entry price", "deposit collection"],
            "weaknesses": ["No approval-first", "no bilingual", "no referral"],
        },
        "inkflow_studio": {
            "pricing": "$39-179/month",
            "strengths": ["Built by tattoo artist", "comprehensive"],
            "weaknesses": ["$39 solo price", "not launched"],
        },
        "keep_the_fees": {
            "pricing": "$35/month",
            "strengths": ["Studio compliance"],
            "weaknesses": ["Not for solo artists", "higher price"],
        },
    },
    "vision": "Become the standard booking infrastructure for independent tattoo artists worldwide.",
}

# ========== ANOMALY THRESHOLDS ==========
ANOMALY_NO_BOOKINGS_HOURS = 48
ANOMALY_EMAIL_USAGE_PERCENT = 80
ANOMALY_NO_SIGNUPS_DAYS = 7

# ========== CACHE SETTINGS ==========
GEMINI_CACHE_TTL_SECONDS = 3600  # 1 hour
MAX_CACHE_ENTRIES = 200

# ========== SQLITE DATABASE PATH ==========
SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH", "/app/data/inkflow_local.db")
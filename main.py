from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from config import PRICING, BUSINESS_CONTEXT
from db.firebase_client import get_firebase_client
from engine.analyzer import Analyzer
from engine.ai_enhancer import get_ai_enhancer
from tasks.scheduler import get_scheduler

# ========== APP INIT ==========
app = FastAPI(
    title="InkFlow Backend",
    description="Intelligence layer for the InkFlow ecosystem — AI analytics, scheduled tasks, REST API",
    version="1.0.0",
)

# CORS — allow frontend from anywhere (local use)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== STARTUP / SHUTDOWN ==========
@app.on_event("startup")
async def startup():
    """Initialize Firebase, start scheduler, take initial snapshot."""
    print("🚀 InkFlow Backend starting...")
    try:
        fb = get_firebase_client()
        print("✅ Firebase connected")
    except Exception as e:
        print(f"⚠️ Firebase connection delayed: {e}")

    try:
        scheduler = get_scheduler()
        scheduler.start()
    except Exception as e:
        print(f"⚠️ Scheduler start delayed: {e}")

    print("✅ InkFlow Backend ready")


@app.on_event("shutdown")
async def shutdown():
    """Stop scheduler gracefully."""
    try:
        scheduler = get_scheduler()
        scheduler.stop()
    except Exception:
        pass
    print("👋 InkFlow Backend shutting down")


# ========== REQUEST MODELS ==========
class LeadScoreRequest(BaseModel):
    artist: Dict[str, Any]


class MessageGradeRequest(BaseModel):
    message: str


class GenerateRequest(BaseModel):
    text: str
    provider: str = "gemini"
    model: str = "flash"
    mode: str = "general"
    imageBase64: Optional[str] = None


class ReportRequest(BaseModel):
    reportType: str = "daily"  # daily, weekly, monthly
    provider: Optional[str] = "gemini"
    model: Optional[str] = "pro"


class TriggerTaskRequest(BaseModel):
    taskName: str


# ========== ROOT & HEALTH ==========
@app.get("/")
async def root():
    """Root status endpoint."""
    return {
        "service": "InkFlow Backend",
        "version": "1.0.0",
        "status": "online",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/status")
async def system_status():
    """Full system status with scheduler info."""
    scheduler = get_scheduler()
    return {
        "status": "online",
        "scheduler": scheduler.get_status(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ========== ANALYZE ENDPOINTS ==========
@app.get("/api/v1/analyze/dashboard")
async def analyze_dashboard():
    """
    Full dashboard analysis.
    Returns exact metrics matching frontend expectations.
    """
    try:
        fb = get_firebase_client()
        analyzer = Analyzer()

        artists = fb.get_all_artists()
        bookings = fb.get_all_bookings()
        outreach_logs = fb.get_outreach_logs()
        payments = fb.get_pending_payments()
        whatsapp = fb.get_whatsapp_stats()
        side_hustle = fb.get_side_hustle()
        honeypot = fb.get_honeypot_logs()
        snapshots = fb.get_daily_snapshots()

        result = analyzer.analyze_dashboard(
            artists=artists,
            bookings=bookings,
            outreach_logs=outreach_logs,
            payments=payments,
            whatsapp_stats=whatsapp,
            side_hustle=side_hustle,
            honeypot_logs=honeypot,
            snapshots=snapshots,
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analyze/outreach")
async def analyze_outreach():
    """Outreach performance analysis."""
    try:
        fb = get_firebase_client()
        analyzer = Analyzer()

        outreach_logs = fb.get_outreach_logs(limit=200)
        artists = fb.get_all_artists()

        funnel = analyzer.analyze_funnel(artists, outreach_logs)

        # Per-channel breakdown
        channels = {}
        for log in outreach_logs:
            channel = log.get("channel", "unknown")
            if channel not in channels:
                channels[channel] = {"sent": 0, "replies": 0}
            if log.get("action") == "dm_sent":
                channels[channel]["sent"] += 1
            elif log.get("action") == "reply_received":
                channels[channel]["replies"] += 1

        channel_breakdown = {}
        for ch, counts in channels.items():
            rate = (
                round((counts["replies"] / counts["sent"]) * 100, 1)
                if counts["sent"] > 0
                else 0
            )
            channel_breakdown[ch] = {**counts, "replyRate": rate}

        return {
            "funnel": funnel,
            "channelBreakdown": channel_breakdown,
            "totalOutreach": len(outreach_logs),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analyze/competitors")
async def analyze_competitors():
    """Competitor intelligence from business context."""
    competitors = BUSINESS_CONTEXT.get("competitors", {})
    pricing = PRICING

    result = {}
    for name, data in competitors.items():
        result[name] = {
            **data,
            "ourPricing": pricing,
            "ourAdvantages": [
                "Approval-first booking (unique)",
                "Bilingual forms (unique)",
                "Waitlist feature",
                "Referral program",
                f"Lower entry price (${pricing['standard']}/mo)",
            ],
        }

    return {
        "competitors": result,
        "ourPricing": pricing,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ========== PREDICT ENDPOINTS ==========
@app.get("/api/v1/predict/churn")
async def predict_churn():
    """Get all artists with churn risk assessment."""
    try:
        fb = get_firebase_client()
        analyzer = Analyzer()

        artists = fb.get_all_artists()
        results = []

        for artist in artists:
            risk = analyzer.predict_churn(artist)
            results.append({
                "artistId": artist.get("id"),
                "artistName": artist.get("name", "Unknown"),
                "status": artist.get("status"),
                "tier": artist.get("tier", "standard"),
                **risk,
            })

        # Sort by risk score descending
        results.sort(key=lambda r: r["riskScore"], reverse=True)

        return {
            "total": len(results),
            "highRisk": sum(1 for r in results if r["risk"] == "High"),
            "mediumRisk": sum(1 for r in results if r["risk"] == "Medium"),
            "lowRisk": sum(1 for r in results if r["risk"] == "Low"),
            "artists": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict/lead-score")
async def score_lead(request: LeadScoreRequest):
    """Score a single lead."""
    try:
        analyzer = Analyzer()
        result = analyzer.score_lead(request.artist)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict/response")
async def grade_message(request: MessageGradeRequest):
    """Grade an outreach message."""
    try:
        analyzer = Analyzer()
        result = analyzer.grade_message(request.message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== GENERATE ENDPOINTS ==========
@app.post("/api/v1/generate/insights")
async def generate_insights(request: GenerateRequest):
    """
    Generate AI insights with multi-provider failover.
    Supports text and image (vision) inputs.
    """
    try:
        enhancer = get_ai_enhancer()

        # Get current dashboard data for prompt enrichment
        dashboard_data = None
        try:
            fb = get_firebase_client()
            analyzer = Analyzer()
            dashboard_data = analyzer.analyze_dashboard(
                artists=fb.get_all_artists(),
                bookings=fb.get_all_bookings(),
                outreach_logs=fb.get_outreach_logs(limit=50),
                payments=fb.get_pending_payments(),
                whatsapp_stats=fb.get_whatsapp_stats(),
                side_hustle=fb.get_side_hustle(),
                honeypot_logs=fb.get_honeypot_logs(limit=10),
                snapshots=fb.get_daily_snapshots(limit=7),
            )
        except Exception:
            pass  # Dashboard data is optional

        result = await enhancer.generate_insight(
            user_text=request.text,
            provider=request.provider,
            model=request.model,
            mode=request.mode,
            dashboard_data=dashboard_data,
            image_base64=request.imageBase64,
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate/message")
async def generate_message(
    artist_name: str = Body(...),
    channel: str = Body("Instagram DM"),
    provider: str = Body("gemini"),
    model: str = Body("flash"),
):
    """Generate an outreach message for a specific artist."""
    try:
        enhancer = get_ai_enhancer()
        prompt = (
            f"Draft a personalized outreach message for a tattoo artist named {artist_name}. "
            f"Channel: {channel}. "
            f"Use the AIDA or PAS framework. Keep it under 250 characters. "
            f"Include a genuine compliment about their work. "
            f"End with a low-pressure question. No links on first contact."
        )

        result = await enhancer.generate_insight(
            user_text=prompt,
            provider=provider,
            model=model,
            mode="outreach",
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate/report")
async def generate_report(request: ReportRequest):
    """Generate a daily/weekly/monthly report using AI."""
    try:
        fb = get_firebase_client()
        analyzer = Analyzer()

        # Fetch data
        artists = fb.get_all_artists()
        bookings = fb.get_all_bookings()
        outreach_logs = fb.get_outreach_logs(limit=500)
        payments = fb.get_pending_payments()
        snapshots = fb.get_daily_snapshots(limit=30)

        dashboard = analyzer.analyze_dashboard(
            artists=artists,
            bookings=bookings,
            outreach_logs=outreach_logs,
            payments=payments,
            whatsapp_stats=fb.get_whatsapp_stats(),
            side_hustle=fb.get_side_hustle(),
            honeypot_logs=fb.get_honeypot_logs(),
            snapshots=snapshots,
        )

        enhancer = get_ai_enhancer()
        prompt = (
            f"Generate a {request.reportType} business report for InkFlow based on this data:\n"
            f"Artists: {dashboard['totalArtists']} total, {dashboard['activeArtists']} active, "
            f"{dashboard['trialArtists']} trial\n"
            f"MRR: ${dashboard['mrr']}\n"
            f"Bookings: {dashboard['totalBookings']} total, {dashboard['bookings7d']} this week\n"
            f"Outreach: {dashboard['dmsSent']} DMs, {dashboard['replyRate']}% reply rate\n"
            f"Health: {dashboard['healthScore']}% ({dashboard['healthLabel']})\n"
            f"Anomalies: {len(dashboard.get('anomalies', []))}\n\n"
            f"Format: Executive summary, key metrics, trends, risks, recommendations."
        )

        result = await enhancer.generate_insight(
            user_text=prompt,
            provider=request.provider or "gemini",
            model=request.model or "pro",
            mode="analyst",
            dashboard_data=dashboard,
        )

        # Save report
        fb.save_report(request.reportType, {
            "prompt": prompt,
            "response": result.get("response"),
            "metrics": dashboard,
        })

        return {
            **result,
            "metrics": dashboard,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== DATA ENDPOINTS ==========
@app.get("/api/v1/data/trends")
async def get_trends():
    """Get metric trends from daily snapshots."""
    try:
        fb = get_firebase_client()
        analyzer = Analyzer()
        snapshots = fb.get_daily_snapshots(limit=30)
        trends = analyzer.detect_trends(snapshots)
        return trends
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/data/cohorts")
async def get_cohorts():
    """Get cohort retention analysis."""
    try:
        fb = get_firebase_client()
        analyzer = Analyzer()
        artists = fb.get_all_artists()
        cohorts = analyzer.analyze_cohorts(artists)
        return {"cohorts": cohorts, "total": len(cohorts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/data/funnels")
async def get_funnels():
    """Get conversion funnel data."""
    try:
        fb = get_firebase_client()
        analyzer = Analyzer()
        artists = fb.get_all_artists()
        outreach_logs = fb.get_outreach_logs(limit=500)
        funnel = analyzer.analyze_funnel(artists, outreach_logs)
        return funnel
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/data/snapshots")
async def get_snapshots(limit: int = Query(30, ge=1, le=90)):
    """Get daily snapshots."""
    try:
        fb = get_firebase_client()
        snapshots = fb.get_daily_snapshots(limit=limit)
        return {"snapshots": snapshots, "count": len(snapshots)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== TASK ENDPOINTS ==========
@app.get("/api/v1/tasks/status")
async def tasks_status():
    """Get scheduler status."""
    scheduler = get_scheduler()
    return scheduler.get_status()


@app.post("/api/v1/tasks/trigger")
async def trigger_task(request: TriggerTaskRequest):
    """Manually trigger a scheduled task."""
    scheduler = get_scheduler()
    result = scheduler.trigger_task(request.taskName)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
    return result


# ========== VISION ENDPOINT ==========
@app.post("/api/v1/generate/vision")
async def analyze_vision(request: GenerateRequest):
    """Analyze an image using Gemini Vision."""
    if not request.imageBase64:
        raise HTTPException(status_code=400, detail="imageBase64 is required")

    try:
        enhancer = get_ai_enhancer()
        result = await enhancer.analyze_image(
            image_base64=request.imageBase64,
            prompt_text=request.text or "Describe this image in detail.",
        )
        if result:
            return {"response": result, "provider": "gemini", "success": True}
        return {"response": None, "provider": "gemini", "success": False}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
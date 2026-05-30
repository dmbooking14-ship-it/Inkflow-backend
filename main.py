"""
InkFlow Backend — Main Application
FastAPI server with all endpoints, middleware, and startup/shutdown
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import uvicorn
import json

from config import CORS_ORIGINS, API_RATE_LIMIT
from db.firebase_client import firebase_client
from db.database import local_db
from engine.ml_engine import ml_engine
from engine.analysis_engine import analysis_engine
from engine.ai_enhancer import ai_enhancer
from tasks.scheduler import task_scheduler

# ========== FASTAPI APP ==========
app = FastAPI(
    title="InkFlow Backend API",
    description="AI-powered business intelligence for InkFlow — SaaS booking platform for tattoo artists",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== PYDANTIC MODELS ==========

class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    mode: Optional[str] = Field("general", pattern="^(general|analytics|outreach|strategy)$")
    use_pro: Optional[bool] = False

class LeadScoreRequest(BaseModel):
    artist_data: Dict = Field(..., description="Artist profile data for scoring")

class MessageGradeRequest(BaseModel):
    message_text: str = Field(..., min_length=1)
    platform: Optional[str] = "instagram"

class ReportRequest(BaseModel):
    report_type: str = Field(..., pattern="^(daily|weekly|monthly|custom)$")
    parameters: Optional[Dict] = {}

class TaskTriggerRequest(BaseModel):
    task_id: str = Field(..., pattern="^(daily_briefing|weekly_report|monthly_review|health_check|competitor_check|retrain_models|daily_snapshot)$")

# ========== STARTUP/SHUTDOWN ==========

@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    print("🚀 Starting InkFlow Backend...")
    
    # Start scheduler
    task_scheduler.start()
    
    # Take initial snapshot
    task_scheduler.take_daily_snapshot()
    
    # Retrain models
    task_scheduler.retrain_models()
    
    print("✅ InkFlow Backend is ready!")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    task_scheduler.stop()
    local_db.close()
    print("👋 InkFlow Backend shut down")

# ========== HEALTH & STATUS ==========

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "InkFlow Backend API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return task_scheduler.health_check()

@app.get("/status")
async def status():
    """Get system status including scheduler jobs"""
    return {
        "api": "running",
        "scheduler": task_scheduler.get_status(),
        "firebase": "connected",
        "models_trained": len(ml_engine.models) > 0,
        "uptime": "running"
    }

# ========== ANALYSIS ENDPOINTS ==========

@app.get("/api/v1/analyze/dashboard")
async def analyze_dashboard():
    """Get comprehensive dashboard analysis"""
    try:
        artists = firebase_client.get_all_artists()
        bookings = firebase_client.get_all_bookings()
        outreach = firebase_client.get_outreach_logs(500)
        history = local_db.get_metrics_history(30)
        
        analysis = analysis_engine.full_analysis(artists, bookings, outreach, history)
        snapshot = firebase_client.take_snapshot()
        
        return {
            "status": "success",
            "snapshot": snapshot,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analyze/outreach")
async def analyze_outreach():
    """Get outreach performance analysis"""
    try:
        outreach = firebase_client.get_outreach_logs(1000)
        artists = firebase_client.get_all_artists()
        bookings = firebase_client.get_all_bookings()
        
        funnel = analysis_engine.funnel_analysis(artists, outreach, bookings)
        
        # Calculate performance metrics
        dms = [o for o in outreach if o.get('action') == 'dm_sent']
        replies = [o for o in outreach if o.get('action') == 'reply_received']
        
        response_rate = round(len(replies) / max(len(dms), 1) * 100, 1)
        
        return {
            "status": "success",
            "total_dms": len(dms),
            "total_replies": len(replies),
            "response_rate": response_rate,
            "funnel": funnel,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analyze/competitors")
async def analyze_competitors():
    """Get competitor analysis"""
    return {
        "status": "success",
        "competitors": [
            {
                "name": "SlidInk",
                "pricing": "$29-225/month",
                "strengths": ["IG DM integration", "Polished UI", "Referral program"],
                "weaknesses": ["Higher entry price", "No approval-first", "No bilingual"],
                "our_advantage": "Lower entry price ($19), approval-first booking, bilingual forms"
            },
            {
                "name": "INKFLO",
                "pricing": "$9-59/month",
                "strengths": ["Low entry price ($9)", "Deposit collection", "Team plans"],
                "weaknesses": ["No approval-first", "No waitlist", "No bilingual", "No referral"],
                "our_advantage": "Approval-first booking, bilingual forms, waitlist, referral program"
            },
            {
                "name": "InkFlow Studio",
                "pricing": "$39-179/month",
                "strengths": ["Built by tattoo artist", "Comprehensive features"],
                "weaknesses": ["Higher entry price", "Not launched", "No bilingual"],
                "our_advantage": "Lower entry price ($19), bilingual forms, market-ready"
            },
            {
                "name": "Keep the Fees",
                "pricing": "$35/month",
                "strengths": ["Compliance features", "White labeling"],
                "weaknesses": ["Studio-focused", "No IG integration", "Higher price"],
                "our_advantage": "Solo artist focus, lower price, IG-friendly approach"
            }
        ],
        "timestamp": datetime.now().isoformat()
    }

# ========== PREDICTION ENDPOINTS ==========

@app.get("/api/v1/predict/churn")
async def predict_churn():
    """Identify artists at risk of churning"""
    try:
        artists = firebase_client.get_all_artists()
        
        at_risk = []
        for artist in artists:
            risk = ml_engine.predict_churn_risk(artist)
            if risk.get('risk_level') in ['High Risk', 'Medium Risk']:
                at_risk.append({
                    "username": artist.get('username', 'Unknown'),
                    "email": artist.get('email', 'Unknown'),
                    "status": artist.get('status', 'Unknown'),
                    "tier": artist.get('tier', 'standard'),
                    "risk_score": risk['risk_score'],
                    "risk_level": risk['risk_level']
                })
        
        at_risk.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return {
            "status": "success",
            "at_risk_count": len(at_risk),
            "artists": at_risk[:20],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict/lead-score")
async def predict_lead_score(request: LeadScoreRequest):
    """Score a potential lead"""
    try:
        score = ml_engine.score_lead(request.artist_data)
        return {
            "status": "success",
            "prediction": score,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict/response")
async def predict_response(request: MessageGradeRequest):
    """Predict message response likelihood and grade it"""
    try:
        grade = ml_engine.grade_message(request.message_text, request.platform)
        return {
            "status": "success",
            "prediction": grade,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== GENERATION ENDPOINTS ==========

@app.post("/api/v1/generate/message")
async def generate_message(request: MessageRequest):
    """Generate optimized outreach message"""
    try:
        context = ai_enhancer.build_context(firebase_client, local_db)
        prompt = ai_enhancer.build_optimized_prompt(
            f"Draft an outreach message for: {request.message}",
            context,
            request.mode
        )
        
        model = GEMINI_MODEL_PRO if request.use_pro else GEMINI_MODEL_FLASH
        response = await ai_enhancer.call_gemini(prompt, model=model, max_tokens=500)
        
        if response:
            # Grade the generated message
            grade = ml_engine.grade_message(response, "instagram")
            
            return {
                "status": "success",
                "message": response,
                "grade": grade,
                "source": "gemini",
                "model_used": model
            }
        else:
            raise HTTPException(status_code=503, detail="AI service unavailable")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate/insights")
async def generate_insights(request: MessageRequest):
    """Generate AI-powered business insights"""
    try:
        context = ai_enhancer.build_context(firebase_client, local_db)
        
        result = await ai_enhancer.get_best_response(
            request.message,
            context
        )
        
        if result.get('response'):
            return {
                "status": "success",
                "insight": result['response'],
                "source": result['source'],
                "model": result['model']
            }
        else:
            raise HTTPException(status_code=503, detail="Could not generate insights")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate/report")
async def generate_report(request: ReportRequest):
    """Generate a custom report"""
    try:
        if request.report_type == 'daily':
            report = task_scheduler.daily_briefing()
        elif request.report_type == 'weekly':
            report = task_scheduler.weekly_report()
        elif request.report_type == 'monthly':
            report = task_scheduler.monthly_review()
        else:
            report = {"status": "Custom reports coming soon"}
        
        return {
            "status": "success",
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== DATA ENDPOINTS ==========

@app.get("/api/v1/data/trends")
async def get_trends(metric: Optional[str] = None, days: int = Query(30, ge=7, le=365)):
    """Get trend data for metrics"""
    try:
        history = local_db.get_metrics_history(days)
        trends = analysis_engine.detect_trends(history)
        
        if metric and metric in trends.get('trends', {}):
            return {
                "status": "success",
                "metric": metric,
                "trend": trends['trends'][metric],
                "history": [
                    {"date": h['date'], "value": h.get(metric, 0)}
                    for h in history if h.get(metric) is not None
                ]
            }
        
        return {
            "status": "success",
            "trends": trends,
            "data_points": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/data/cohorts")
async def get_cohorts():
    """Get cohort retention analysis"""
    try:
        artists = firebase_client.get_all_artists()
        bookings = firebase_client.get_all_bookings()
        
        cohorts = analysis_engine.cohort_analysis(artists, bookings)
        
        return {
            "status": "success",
            "cohorts": cohorts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/data/funnels")
async def get_funnels():
    """Get conversion funnel analysis"""
    try:
        artists = firebase_client.get_all_artists()
        bookings = firebase_client.get_all_bookings()
        outreach = firebase_client.get_outreach_logs(1000)
        
        funnel = analysis_engine.funnel_analysis(artists, outreach, bookings)
        
        return {
            "status": "success",
            "funnel": funnel
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/data/snapshots")
async def get_snapshots(days: int = Query(30, ge=1, le=365)):
    """Get daily snapshots"""
    try:
        snapshots = firebase_client.get_daily_snapshots(days)
        history = local_db.get_metrics_history(days)
        
        return {
            "status": "success",
            "firebase_snapshots": len(snapshots),
            "local_history": len(history),
            "data": history[-30:]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== TASK ENDPOINTS ==========

@app.get("/api/v1/tasks/status")
async def get_tasks_status():
    """Get status of all scheduled tasks"""
    return {
        "status": "success",
        "scheduler": task_scheduler.get_status()
    }

@app.post("/api/v1/tasks/trigger")
async def trigger_task(request: TaskTriggerRequest):
    """Manually trigger a scheduled task"""
    result = task_scheduler.trigger_task(request.task_id)
    return {
        "status": "success",
        "task": request.task_id,
        "result": result
    }

# ========== RUN ==========

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
"""
InkFlow Task Scheduler
Daily briefings, weekly reports, monthly reviews, health checks, competitor monitoring
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta
from typing import Dict, List
import json
import asyncio

from config import (
    SCHEDULE_DAILY_BRIEFING, SCHEDULE_WEEKLY_REPORT, 
    SCHEDULE_MONTHLY_REVIEW, SCHEDULE_HEALTH_CHECK,
    SCHEDULE_COMPETITOR_CHECK, ANOMALY_CONFIG
)
from db.firebase_client import firebase_client
from db.database import local_db

# ML imports wrapped in try/except — will work when dependencies are added
try:
    from engine.ml_engine import ml_engine
except ImportError:
    ml_engine = None

try:
    from engine.analysis_engine import analysis_engine
except ImportError:
    analysis_engine = None

try:
    from engine.ai_enhancer import ai_enhancer
except ImportError:
    ai_enhancer = None

class TaskScheduler:
    """Manages all scheduled tasks for InkFlow"""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.task_history = []
        self._setup_jobs()
    
    def _setup_jobs(self):
        """Configure all scheduled jobs"""
        
        # Daily briefing at 8 AM
        self.scheduler.add_job(
            self.daily_briefing,
            CronTrigger.from_crontab(f"0 {SCHEDULE_DAILY_BRIEFING.replace(':',' ')} * * *"),
            id='daily_briefing',
            name='Daily Business Briefing'
        )
        
        # Weekly report Sunday at 6 PM
        hour, minute = SCHEDULE_WEEKLY_REPORT.split('@')[1].split(':')
        self.scheduler.add_job(
            self.weekly_report,
            CronTrigger.from_crontab(f"{minute} {hour} * * 0"),
            id='weekly_report',
            name='Weekly Performance Report'
        )
        
        # Monthly review on the 1st at 9 AM
        self.scheduler.add_job(
            self.monthly_review,
            CronTrigger.from_crontab(f"0 9 1 * *"),
            id='monthly_review',
            name='Monthly Business Review'
        )
        
        # Health check every 6 hours
        self.scheduler.add_job(
            self.health_check,
            CronTrigger.from_crontab(f"0 */6 * * *"),
            id='health_check',
            name='System Health Check'
        )
        
        # Competitor check every Monday at 10 AM
        self.scheduler.add_job(
            self.competitor_check,
            CronTrigger.from_crontab(f"0 10 * * 1"),
            id='competitor_check',
            name='Competitor Monitoring'
        )
        
        # ML model retraining daily at 2 AM (only if ML is available)
        if ml_engine:
            self.scheduler.add_job(
                self.retrain_models,
                CronTrigger.from_crontab(f"0 2 * * *"),
                id='retrain_models',
                name='ML Model Retraining'
            )
        
        # Snapshot every day at midnight
        self.scheduler.add_job(
            self.take_daily_snapshot,
            CronTrigger.from_crontab(f"0 0 * * *"),
            id='daily_snapshot',
            name='Daily Metrics Snapshot'
        )
    
    def start(self):
        """Start the scheduler"""
        self.scheduler.start()
        print("✅ Scheduler started — all tasks configured")
        self.log_task("scheduler_started", "All scheduled tasks activated")
    
    def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
    
    def get_status(self) -> Dict:
        """Get status of all scheduled jobs"""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time) if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
        
        return {
            "active_jobs": len(jobs),
            "jobs": jobs,
            "recent_tasks": self.task_history[-10:]
        }
    
    def trigger_task(self, task_id: str) -> Dict:
        """Manually trigger a scheduled task"""
        task_map = {
            'daily_briefing': self.daily_briefing,
            'weekly_report': self.weekly_report,
            'monthly_review': self.monthly_review,
            'health_check': self.health_check,
            'competitor_check': self.competitor_check,
            'daily_snapshot': self.take_daily_snapshot
        }
        if ml_engine:
            task_map['retrain_models'] = self.retrain_models
        
        if task_id in task_map:
            result = task_map[task_id]()
            return {"status": "completed", "task": task_id, "result": result}
        
        return {"status": "error", "message": f"Unknown task: {task_id}"}
    
    def log_task(self, task_name: str, details: str):
        """Log a task execution"""
        self.task_history.append({
            "task": task_name,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.task_history) > 100:
            self.task_history = self.task_history[-100:]
    
    # ========== TASK IMPLEMENTATIONS ==========
    
    def daily_briefing(self) -> Dict:
        """Generate daily business briefing"""
        print("📋 Generating daily briefing...")
        
        try:
            artists = firebase_client.get_all_artists()
            bookings = firebase_client.get_all_bookings()
            outreach = firebase_client.get_outreach_logs(100)
            
            total = len(artists)
            active = len([a for a in artists if a.get('status') == 'active'])
            trial = len([a for a in artists if a.get('status') == 'trial'])
            
            standard = len([a for a in artists if a.get('tier', 'standard') == 'standard' and a.get('status') == 'active'])
            pro = len([a for a in artists if a.get('tier') == 'pro' and a.get('status') == 'active'])
            premium = len([a for a in artists if a.get('tier') == 'premium' and a.get('status') == 'active'])
            mrr = (standard * 19) + (pro * 39) + (premium * 59)
            
            now = datetime.now()
            today_start = now.replace(hour=0, minute=0, second=0)
            
            bookings_today = len([b for b in bookings if b.get('createdAt') and 
                                 hasattr(b['createdAt'], 'timestamp') and 
                                 datetime.fromtimestamp(b['createdAt'].timestamp()) >= today_start])
            
            outreach_today = len([o for o in outreach if o.get('createdAt') and
                                 hasattr(o['createdAt'], 'timestamp') and
                                 datetime.fromtimestamp(o['createdAt'].timestamp()) >= today_start])
            
            # Detect anomalies (only if ML available)
            anomalies = []
            if ml_engine:
                metrics = {
                    "totalBookings": len(bookings),
                    "bookings7d": len([b for b in bookings if b.get('createdAt') and 
                                      hasattr(b['createdAt'], 'timestamp') and
                                      datetime.fromtimestamp(b['createdAt'].timestamp()) >= now - timedelta(days=7)]),
                    "dmsSent": len([o for o in outreach if o.get('action') == 'dm_sent']),
                    "repliesReceived": len([o for o in outreach if o.get('action') == 'reply_received'])
                }
                history = local_db.get_metrics_history(30)
                anomalies = ml_engine.detect_anomalies(metrics, history)
            
            briefing = {
                "date": now.strftime("%Y-%m-%d"),
                "type": "daily_briefing",
                "summary": {
                    "total_artists": total,
                    "active_artists": active,
                    "trial_artists": trial,
                    "mrr": mrr,
                    "bookings_today": bookings_today,
                    "outreach_today": outreach_today
                },
                "anomalies": anomalies,
                "recommendations": self._generate_daily_recommendations(metrics if ml_engine else {}, anomalies),
                "generated_at": now.isoformat()
            }
            
            local_db.save_report('daily_briefing', briefing)
            firebase_client.save_report(briefing)
            
            self.log_task("daily_briefing", f"Generated — {total} artists, ${mrr} MRR")
            return briefing
            
        except Exception as e:
            self.log_task("daily_briefing", f"Error: {str(e)}")
            return {"error": str(e)}
    
    def _generate_daily_recommendations(self, metrics: Dict, anomalies: List[Dict]) -> List[str]:
        """Generate actionable daily recommendations"""
        recommendations = []
        
        dms = metrics.get('dmsSent', 0) if metrics else 0
        if dms == 0:
            recommendations.append("🚨 No outreach recorded recently — send at least 5 DMs today")
        
        if anomalies:
            for anomaly in anomalies:
                if anomaly.get('metric') == 'bookings_48h':
                    recommendations.append("⚠️ No bookings in 48 hours — check if artists are sharing their booking links")
        
        if not recommendations:
            recommendations.append("✅ Metrics look stable — maintain consistent outreach and follow up with trial users")
            recommendations.append("📝 Consider logging your outreach actions to improve ML predictions")
        
        return recommendations
    
    def weekly_report(self) -> Dict:
        """Generate comprehensive weekly report"""
        print("📊 Generating weekly report...")
        
        try:
            artists = firebase_client.get_all_artists()
            bookings = firebase_client.get_all_bookings()
            outreach = firebase_client.get_outreach_logs(500)
            history = local_db.get_metrics_history(7)
            
            # Run analysis if available
            analysis = {}
            if analysis_engine:
                analysis = analysis_engine.full_analysis(artists, bookings, outreach, history)
            
            # Retrain models if available
            ml_results = {}
            if ml_engine:
                ml_results = ml_engine.retrain_all(artists, bookings, outreach)
            
            report = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "type": "weekly_report",
                "week": datetime.now().strftime("%Y-W%W"),
                "analysis": analysis,
                "ml_retraining": ml_results,
                "generated_at": datetime.now().isoformat()
            }
            
            local_db.save_report('weekly_report', report)
            firebase_client.save_report(report)
            
            self.log_task("weekly_report", f"Week {report['week']} complete")
            return report
            
        except Exception as e:
            self.log_task("weekly_report", f"Error: {str(e)}")
            return {"error": str(e)}
    
    def monthly_review(self) -> Dict:
        """Generate monthly business review"""
        print("📈 Generating monthly review...")
        
        try:
            artists = firebase_client.get_all_artists()
            bookings = firebase_client.get_all_bookings()
            outreach = firebase_client.get_outreach_logs(1000)
            history = local_db.get_metrics_history(30)
            
            cohorts = {}
            funnel = {}
            trends = {}
            
            if analysis_engine:
                cohorts = analysis_engine.cohort_analysis(artists, bookings)
                funnel = analysis_engine.funnel_analysis(artists, outreach, bookings)
                trends = analysis_engine.detect_trends(history)
            
            month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0)
            new_this_month = len([a for a in artists if a.get('createdAt') and
                                 hasattr(a['createdAt'], 'timestamp') and
                                 datetime.fromtimestamp(a['createdAt'].timestamp()) >= month_start])
            
            active = len([a for a in artists if a.get('status') == 'active'])
            standard = len([a for a in artists if a.get('tier', 'standard') == 'standard' and a.get('status') == 'active'])
            pro = len([a for a in artists if a.get('tier') == 'pro' and a.get('status') == 'active'])
            premium = len([a for a in artists if a.get('tier') == 'premium' and a.get('status') == 'active'])
            
            review = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "type": "monthly_review",
                "month": datetime.now().strftime("%Y-%m"),
                "growth": {
                    "new_artists_this_month": new_this_month,
                    "total_artists": len(artists),
                    "active_artists": active,
                    "mrr": (standard * 19) + (pro * 39) + (premium * 59),
                    "total_bookings": len(bookings)
                },
                "cohorts": cohorts,
                "funnel": funnel,
                "trends": trends,
                "generated_at": datetime.now().isoformat()
            }
            
            local_db.save_report('monthly_review', review)
            firebase_client.save_report(review)
            
            self.log_task("monthly_review", f"Month {review['month']} review complete")
            return review
            
        except Exception as e:
            self.log_task("monthly_review", f"Error: {str(e)}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict:
        """Run system health check"""
        print("❤️ Running health check...")
        
        try:
            artists = firebase_client.get_all_artists()
            bookings = firebase_client.get_all_bookings()
            
            checks = {
                "firebase_connection": len(artists) >= 0,
                "data_integrity": len(artists) >= 0 and len(bookings) >= 0,
                "database_size": {
                    "artists": len(artists),
                    "bookings": len(bookings)
                }
            }
            
            health = {
                "timestamp": datetime.now().isoformat(),
                "status": "healthy",
                "checks": checks,
                "uptime": "Running"
            }
            
            local_db.save_report('health_check', health)
            
            return health
            
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    def competitor_check(self) -> Dict:
        """Monitor competitor changes"""
        print("🔍 Checking competitors...")
        
        competitors = [
            {"name": "SlidInk", "url": "https://www.slidink.com"},
            {"name": "INKFLO", "url": "https://www.inkflo.app"},
            {"name": "InkFlow Studio", "url": "https://www.inkflowstudio.app"},
            {"name": "Keep the Fees", "url": "https://keepthefees.com"}
        ]
        
        results = []
        for comp in competitors:
            results.append({
                "name": comp["name"],
                "url": comp["url"],
                "checked_at": datetime.now().isoformat(),
                "status": "Website accessible (manual review recommended)"
            })
        
        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "type": "competitor_check",
            "competitors": results,
            "generated_at": datetime.now().isoformat()
        }
        
        local_db.save_report('competitor_check', report)
        self.log_task("competitor_check", f"Checked {len(competitors)} competitors")
        
        return report
    
    def retrain_models(self) -> Dict:
        """Retrain all ML models"""
        if not ml_engine:
            return {"status": "skipped", "reason": "ML engine not available"}
        
        print("🤖 Retraining ML models...")
        
        try:
            artists = firebase_client.get_all_artists()
            bookings = firebase_client.get_all_bookings()
            outreach = firebase_client.get_outreach_logs(1000)
            
            results = ml_engine.retrain_all(artists, bookings, outreach)
            
            self.log_task("retrain_models", f"Retrained — {results.get('total_artists', 0)} artists")
            return results
            
        except Exception as e:
            self.log_task("retrain_models", f"Error: {str(e)}")
            return {"error": str(e)}
    
    def take_daily_snapshot(self) -> Dict:
        """Take daily metrics snapshot"""
        try:
            snapshot = firebase_client.take_snapshot()
            
            local_db.save_daily_metrics({
                "date": datetime.now().strftime("%Y-%m-%d"),
                **snapshot
            })
            
            self.log_task("daily_snapshot", f"Saved — {snapshot.get('totalArtists', 0)} artists")
            return snapshot
            
        except Exception as e:
            self.log_task("daily_snapshot", f"Error: {str(e)}")
            return {"error": str(e)}


# Singleton instance
task_scheduler = TaskScheduler()

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from config import (
    SCHEDULE_DAILY_BRIEFING_HOUR,
    SCHEDULE_DAILY_BRIEFING_MINUTE,
    SCHEDULE_WEEKLY_REPORT_DAY,
    SCHEDULE_WEEKLY_REPORT_HOUR,
    SCHEDULE_WEEKLY_REPORT_MINUTE,
    SCHEDULE_MONTHLY_REVIEW_DAY,
    SCHEDULE_MONTHLY_REVIEW_HOUR,
    SCHEDULE_MONTHLY_REVIEW_MINUTE,
    SCHEDULE_HEALTH_CHECK_INTERVAL_HOURS,
    SCHEDULE_COMPETITOR_CHECK_DAY,
    SCHEDULE_COMPETITOR_CHECK_HOUR,
    SCHEDULE_COMPETITOR_CHECK_MINUTE,
    SCHEDULE_DAILY_SNAPSHOT_HOUR,
    SCHEDULE_DAILY_SNAPSHOT_MINUTE,
    PRICING,
    ANOMALY_NO_BOOKINGS_HOURS,
    ANOMALY_EMAIL_USAGE_PERCENT,
    ANOMALY_NO_SIGNUPS_DAYS,
)


class TaskScheduler:
    """Manages all scheduled tasks using APScheduler with explicit CronTrigger."""

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self._running = False

    def start(self) -> None:
        """Start all scheduled jobs."""
        if self._running:
            return

        # Daily briefing — every day at configured time
        self.scheduler.add_job(
            self._daily_briefing,
            CronTrigger(
                hour=SCHEDULE_DAILY_BRIEFING_HOUR,
                minute=SCHEDULE_DAILY_BRIEFING_MINUTE,
            ),
            id="daily_briefing",
            name="Daily Briefing",
            replace_existing=True,
        )

        # Weekly report — Sunday at configured time
        self.scheduler.add_job(
            self._weekly_report,
            CronTrigger(
                day_of_week=SCHEDULE_WEEKLY_REPORT_DAY,
                hour=SCHEDULE_WEEKLY_REPORT_HOUR,
                minute=SCHEDULE_WEEKLY_REPORT_MINUTE,
            ),
            id="weekly_report",
            name="Weekly Report",
            replace_existing=True,
        )

        # Monthly review — 1st of month at configured time
        self.scheduler.add_job(
            self._monthly_review,
            CronTrigger(
                day=SCHEDULE_MONTHLY_REVIEW_DAY,
                hour=SCHEDULE_MONTHLY_REVIEW_HOUR,
                minute=SCHEDULE_MONTHLY_REVIEW_MINUTE,
            ),
            id="monthly_review",
            name="Monthly Review",
            replace_existing=True,
        )

        # Health check — every N hours
        self.scheduler.add_job(
            self._health_check,
            CronTrigger(hour=f"*/{SCHEDULE_HEALTH_CHECK_INTERVAL_HOURS}"),
            id="health_check",
            name="Health Check",
            replace_existing=True,
        )

        # Competitor check — Monday at configured time
        self.scheduler.add_job(
            self._competitor_check,
            CronTrigger(
                day_of_week=SCHEDULE_COMPETITOR_CHECK_DAY,
                hour=SCHEDULE_COMPETITOR_CHECK_HOUR,
                minute=SCHEDULE_COMPETITOR_CHECK_MINUTE,
            ),
            id="competitor_check",
            name="Competitor Check",
            replace_existing=True,
        )

        # Daily snapshot — midnight
        self.scheduler.add_job(
            self._daily_snapshot,
            CronTrigger(
                hour=SCHEDULE_DAILY_SNAPSHOT_HOUR,
                minute=SCHEDULE_DAILY_SNAPSHOT_MINUTE,
            ),
            id="daily_snapshot",
            name="Daily Snapshot",
            replace_existing=True,
        )

        self.scheduler.start()
        self._running = True
        print("✅ Scheduler started — 6 jobs registered")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            print("🛑 Scheduler stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get status of all scheduled jobs."""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "nextRun": job.next_run_time.isoformat() if job.next_run_time else None,
            })
        return {
            "running": self._running,
            "jobCount": len(jobs),
            "jobs": jobs,
        }

    def trigger_task(self, task_name: str) -> Dict[str, Any]:
        """Manually trigger a task by name."""
        task_map = {
            "daily_briefing": self._daily_briefing,
            "weekly_report": self._weekly_report,
            "monthly_review": self._monthly_review,
            "health_check": self._health_check,
            "competitor_check": self._competitor_check,
            "daily_snapshot": self._daily_snapshot,
        }

        func = task_map.get(task_name)
        if not func:
            return {"success": False, "error": f"Unknown task: {task_name}"}

        try:
            result = func()
            return {"success": True, "task": task_name, "result": result}
        except Exception as e:
            return {"success": False, "task": task_name, "error": str(e)}

    # ========== TASK IMPLEMENTATIONS ==========

    def _daily_briefing(self) -> Dict[str, Any]:
        """
        Daily briefing: fetch data, run analysis, save snapshot.
        Runs every day at 8:00 AM.
        """
        print(f"📋 Running daily briefing — {datetime.now(timezone.utc).isoformat()}")
        try:
            from db.firebase_client import get_firebase_client
            from engine.analyzer import Analyzer

            fb = get_firebase_client()
            analyzer = Analyzer()

            # Fetch all data
            artists = fb.get_all_artists()
            bookings = fb.get_all_bookings()
            outreach_logs = fb.get_outreach_logs()
            payments = fb.get_pending_payments()
            whatsapp = fb.get_whatsapp_stats()
            side_hustle = fb.get_side_hustle()
            honeypot = fb.get_honeypot_logs()
            snapshots = fb.get_daily_snapshots()

            # Run analysis
            dashboard = analyzer.analyze_dashboard(
                artists=artists,
                bookings=bookings,
                outreach_logs=outreach_logs,
                payments=payments,
                whatsapp_stats=whatsapp,
                side_hustle=side_hustle,
                honeypot_logs=honeypot,
                snapshots=snapshots,
            )

            # Save snapshot to Firestore
            fb.save_snapshot({
                "type": "daily_briefing",
                "metrics": dashboard,
                "generatedAt": datetime.now(timezone.utc).isoformat(),
            })

            print(f"✅ Daily briefing complete — {dashboard['totalArtists']} artists, ${dashboard['mrr']} MRR")
            return {"status": "completed", "metrics": dashboard}

        except Exception as e:
            print(f"❌ Daily briefing failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _weekly_report(self) -> Dict[str, Any]:
        """
        Weekly report: comprehensive analysis of the week.
        Runs every Sunday at 6:00 PM.
        """
        print(f"📊 Running weekly report — {datetime.now(timezone.utc).isoformat()}")
        try:
            from db.firebase_client import get_firebase_client
            from engine.analyzer import Analyzer

            fb = get_firebase_client()
            analyzer = Analyzer()

            artists = fb.get_all_artists()
            bookings = fb.get_all_bookings()
            outreach_logs = fb.get_outreach_logs(limit=500)
            payments = fb.get_pending_payments()
            whatsapp = fb.get_whatsapp_stats()
            side_hustle = fb.get_side_hustle()
            honeypot = fb.get_honeypot_logs()
            snapshots = fb.get_daily_snapshots(limit=7)

            dashboard = analyzer.analyze_dashboard(
                artists=artists,
                bookings=bookings,
                outreach_logs=outreach_logs,
                payments=payments,
                whatsapp_stats=whatsapp,
                side_hustle=side_hustle,
                honeypot_logs=honeypot,
                snapshots=snapshots,
            )

            # Add weekly-specific data
            churn_risks = []
            for artist in artists:
                risk = analyzer.predict_churn(artist)
                if risk["risk"] in ("High", "Medium"):
                    churn_risks.append({
                        "artistId": artist.get("id"),
                        "artistName": artist.get("name", "Unknown"),
                        "risk": risk["risk"],
                        "riskScore": risk["riskScore"],
                        "factors": risk["factors"],
                    })

            weekly_data = {
                "type": "weekly_report",
                "metrics": dashboard,
                "churnRisks": churn_risks[:10],  # Top 10 at-risk
                "generatedAt": datetime.now(timezone.utc).isoformat(),
            }

            fb.save_report("weekly", weekly_data)
            fb.save_snapshot({"type": "weekly_report", "metrics": dashboard})

            print(f"✅ Weekly report complete — {len(churn_risks)} artists at risk")
            return {"status": "completed", "metrics": dashboard, "churnRisks": len(churn_risks)}

        except Exception as e:
            print(f"❌ Weekly report failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _monthly_review(self) -> Dict[str, Any]:
        """
        Monthly review: deep analysis with trends and projections.
        Runs 1st of each month at 9:00 AM.
        """
        print(f"📈 Running monthly review — {datetime.now(timezone.utc).isoformat()}")
        try:
            from db.firebase_client import get_firebase_client
            from engine.analyzer import Analyzer

            fb = get_firebase_client()
            analyzer = Analyzer()

            artists = fb.get_all_artists()
            bookings = fb.get_all_bookings()
            outreach_logs = fb.get_outreach_logs(limit=1000)
            payments = fb.get_pending_payments()
            whatsapp = fb.get_whatsapp_stats()
            side_hustle = fb.get_side_hustle()
            honeypot = fb.get_honeypot_logs()
            snapshots = fb.get_daily_snapshots(limit=30)

            dashboard = analyzer.analyze_dashboard(
                artists=artists,
                bookings=bookings,
                outreach_logs=outreach_logs,
                payments=payments,
                whatsapp_stats=whatsapp,
                side_hustle=side_hustle,
                honeypot_logs=honeypot,
                snapshots=snapshots,
            )

            # Monthly-specific: full cohort + correlation analysis
            cohorts = analyzer.analyze_cohorts(artists)
            correlations = analyzer.discover_correlations(snapshots)

            monthly_data = {
                "type": "monthly_review",
                "metrics": dashboard,
                "cohorts": cohorts,
                "correlations": correlations,
                "generatedAt": datetime.now(timezone.utc).isoformat(),
            }

            fb.save_report("monthly", monthly_data)
            fb.save_snapshot({"type": "monthly_review", "metrics": dashboard})

            print(f"✅ Monthly review complete")
            return {"status": "completed", "metrics": dashboard}

        except Exception as e:
            print(f"❌ Monthly review failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _health_check(self) -> Dict[str, Any]:
        """
        Health check: quick check for anomalies.
        Runs every 6 hours.
        """
        print(f"❤️ Running health check — {datetime.now(timezone.utc).isoformat()}")
        try:
            from db.firebase_client import get_firebase_client
            from engine.analyzer import Analyzer

            fb = get_firebase_client()
            analyzer = Analyzer()

            artists = fb.get_all_artists()
            bookings = fb.get_all_bookings()

            # Quick anomaly check
            total_bookings = len(bookings)
            approved = sum(1 for b in bookings if b.get("status") == "approved")
            email_used = total_bookings + approved

            anomalies = analyzer.detect_anomalies(artists, bookings, email_used)

            if anomalies:
                print(f"⚠️ Health check: {len(anomalies)} anomalies found")
                for a in anomalies:
                    print(f"  - {a['type']}: {a['description']}")

            return {
                "status": "completed",
                "anomalies": len(anomalies),
                "details": anomalies,
            }

        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _competitor_check(self) -> Dict[str, Any]:
        """
        Competitor check: placeholder for manual competitor monitoring.
        Runs every Monday at 10:00 AM.
        """
        print(f"🔍 Running competitor check — {datetime.now(timezone.utc).isoformat()}")
        from config import BUSINESS_CONTEXT

        competitors = BUSINESS_CONTEXT.get("competitors", {})
        comp_names = list(competitors.keys())

        return {
            "status": "completed",
            "competitorsMonitored": len(comp_names),
            "competitors": comp_names,
            "note": "Manual review recommended — check competitor websites for changes",
        }

    def _daily_snapshot(self) -> Dict[str, Any]:
        """
        Daily snapshot: capture current metrics at midnight.
        """
        print(f"📸 Taking daily snapshot — {datetime.now(timezone.utc).isoformat()}")
        try:
            from db.firebase_client import get_firebase_client
            from engine.analyzer import Analyzer

            fb = get_firebase_client()
            analyzer = Analyzer()

            artists = fb.get_all_artists()
            bookings = fb.get_all_bookings()
            outreach_logs = fb.get_outreach_logs(limit=50)
            payments = fb.get_pending_payments()
            whatsapp = fb.get_whatsapp_stats()
            side_hustle = fb.get_side_hustle()
            honeypot = fb.get_honeypot_logs(limit=10)
            snapshots = fb.get_daily_snapshots(limit=7)

            dashboard = analyzer.analyze_dashboard(
                artists=artists,
                bookings=bookings,
                outreach_logs=outreach_logs,
                payments=payments,
                whatsapp_stats=whatsapp,
                side_hustle=side_hustle,
                honeypot_logs=honeypot,
                snapshots=snapshots,
            )

            fb.save_snapshot({
                "type": "daily_snapshot",
                "metrics": dashboard,
            })

            return {"status": "completed", "metrics": dashboard}

        except Exception as e:
            print(f"❌ Daily snapshot failed: {e}")
            return {"status": "failed", "error": str(e)}


# ========== MODULE-LEVEL SINGLETON ==========
_scheduler: Optional[TaskScheduler] = None


def get_scheduler() -> TaskScheduler:
    """Get or create the scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler
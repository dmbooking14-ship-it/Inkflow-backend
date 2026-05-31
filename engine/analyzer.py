File 7: engine/analyzer.py

```python
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple


class Analyzer:
    """Heuristic analysis engine. No pandas, no sklearn — pure Python + numpy."""

    # ========== LEAD SCORING ==========
    @staticmethod
    def score_lead(artist: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a lead based on followers, booking method, status, and activity.
        Returns score 0-1 and quality tier.
        """
        score = 0.0
        reasons = []

        # Followers signal
        followers = artist.get("followers", 0) or 0
        if followers > 10000:
            score += 0.25
            reasons.append("10k+ followers")
        elif followers > 5000:
            score += 0.20
            reasons.append("5k+ followers")
        elif followers > 1000:
            score += 0.10
            reasons.append("1k+ followers")

        # Booking method — DM-based artists have more pain
        booking_method = (artist.get("booking_method") or "").lower()
        if "dm" in booking_method:
            score += 0.25
            reasons.append("books via DMs (high pain)")
        elif "whatsapp" in booking_method:
            score += 0.15
            reasons.append("books via WhatsApp")

        # Status — not contacted yet means fresh opportunity
        status = (artist.get("status") or "").lower()
        if "identified_not_contacted" in status:
            score += 0.15
            reasons.append("not yet contacted")
        elif "messaged_once" in status:
            score += 0.05
            reasons.append("needs follow-up")

        # Priority
        priority = (artist.get("priority") or "").lower()
        if priority == "high":
            score += 0.15
            reasons.append("marked high priority")

        # Notes indicate pain point
        notes = (artist.get("notes") or "").lower()
        if "pain" in notes or "overwhelmed" in notes or "double" in notes:
            score += 0.10
            reasons.append("has clear pain point")

        # Cap at 1.0
        score = min(score, 1.0)

        # Quality tier
        if score >= 0.7:
            quality = "Excellent"
        elif score >= 0.5:
            quality = "Good"
        elif score >= 0.3:
            quality = "Fair"
        else:
            quality = "Low"

        return {
            "score": round(score, 2),
            "quality": quality,
            "reasons": reasons,
        }

    # ========== CHURN PREDICTION ==========
    @staticmethod
    def predict_churn(artist: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict churn risk based on status, trial expiry, and activity.
        Returns risk level and factors.
        """
        risk_score = 0.0
        factors = []

        status = (artist.get("status") or "").lower()

        # Expired = already churned
        if status == "expired":
            return {
                "risk": "High",
                "riskScore": 1.0,
                "factors": ["Already expired"],
                "action": "Re-engage with special offer or feedback request",
            }

        # Trial ending soon
        if status == "trial":
            trial_end = artist.get("trialEndsAt")
            if trial_end:
                try:
                    if isinstance(trial_end, str):
                        trial_end = datetime.fromisoformat(trial_end.replace("Z", "+00:00"))
                    days_left = (trial_end - datetime.now(timezone.utc)).days
                    if days_left <= 3:
                        risk_score += 0.4
                        factors.append(f"Trial ends in {days_left} days")
                    elif days_left <= 7:
                        risk_score += 0.2
                        factors.append(f"Trial ends in {days_left} days")
                except (ValueError, TypeError):
                    pass

        # No recent activity
        last_active = artist.get("lastActiveAt")
        if last_active:
            try:
                if isinstance(last_active, str):
                    last_active = datetime.fromisoformat(last_active.replace("Z", "+00:00"))
                days_inactive = (datetime.now(timezone.utc) - last_active).days
                if days_inactive > 14:
                    risk_score += 0.3
                    factors.append(f"Inactive for {days_inactive} days")
                elif days_inactive > 7:
                    risk_score += 0.15
                    factors.append(f"Inactive for {days_inactive} days")
            except (ValueError, TypeError):
                pass
        else:
            # No activity record at all
            risk_score += 0.1
            factors.append("No activity recorded")

        # No bookings
        booking_count = artist.get("bookingCount", 0) or 0
        if booking_count == 0:
            risk_score += 0.15
            factors.append("No bookings made")

        # Determine risk level
        risk_score = min(risk_score, 1.0)
        if risk_score >= 0.6:
            risk = "High"
            action = "Immediate outreach — call, DM, or personalized email"
        elif risk_score >= 0.3:
            risk = "Medium"
            action = "Send re-engagement email or feature highlight"
        else:
            risk = "Low"
            action = "Regular nurturing"

        return {
            "risk": risk,
            "riskScore": round(risk_score, 2),
            "factors": factors,
            "action": action,
        }

    # ========== MESSAGE GRADING ==========
    @staticmethod
    def grade_message(message: str) -> Dict[str, Any]:
        """
        Grade an outreach message based on length, personalization, questions, and spam triggers.
        Returns grade A-F with suggestions.
        """
        score = 0
        suggestions = []
        lower = message.lower()
        length = len(message)

        # Length scoring
        if 100 <= length <= 350:
            score += 2
        elif 50 <= length < 100:
            score += 1
            suggestions.append("Consider adding a bit more context or a specific compliment")
        elif length < 50:
            suggestions.append("Message is very short — add a personal touch")
        elif length > 500:
            score -= 1
            suggestions.append("Message is long — try to keep under 350 characters")

        # Personalization
        personalization_markers = [
            "saw your", "noticed your", "your work", "your style",
            "your post", "your recent", "you posted", "your portfolio",
        ]
        has_personalization = any(m in lower for m in personalization_markers)
        if has_personalization:
            score += 2
        else:
            suggestions.append("Mention something specific about their work or recent post")

        # Question engagement
        question_count = message.count("?")
        if question_count >= 2:
            score += 1
        elif question_count == 1:
            score += 2
            suggestions.append("Good — one question is ideal")
        else:
            suggestions.append("Include a question to start a conversation")

        # No-pressure language
        no_pressure_markers = [
            "no pressure", "no worries", "either way", "if curious",
            "whenever", "no rush",
        ]
        if any(m in lower for m in no_pressure_markers):
            score += 1

        # Spam triggers — penalize heavily
        spam_triggers = [
            "act now", "limited time", "don't miss out", "exclusive offer",
            "guaranteed", "free money", "click here", "buy now",
            "order now", "special promotion",
        ]
        spam_hits = [t for t in spam_triggers if t in lower]
        if spam_hits:
            score -= len(spam_hits) * 2
            suggestions.append(f"Avoid spam triggers: {', '.join(spam_hits)}")

        # Link penalty (too salesy if early)
        if "http" in lower or "link" in lower:
            score -= 1
            suggestions.append("Consider removing the link on first contact — build rapport first")

        # Determine grade
        if score >= 6:
            grade = "A"
        elif score >= 4:
            grade = "B"
        elif score >= 2:
            grade = "C"
        elif score >= 0:
            grade = "D"
        else:
            grade = "F"

        return {
            "grade": grade,
            "score": score,
            "suggestions": suggestions if suggestions else ["Message looks good"],
            "effectiveness": "High" if grade in ("A", "B") else "Medium" if grade == "C" else "Low",
        }

    # ========== ANOMALY DETECTION ==========
    @staticmethod
    def detect_anomalies(
        artists: List[Dict[str, Any]],
        bookings: List[Dict[str, Any]],
        email_used: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies: no bookings in 48h, email usage >80%, no signups in 7 days.
        """
        anomalies = []
        now = datetime.now(timezone.utc)

        # Check for no bookings in last 48 hours
        recent_bookings = 0
        cutoff_48h = now - timedelta(hours=48)
        for b in bookings:
            created = b.get("createdAt")
            if created:
                try:
                    if hasattr(created, "to_datetime"):
                        created_dt = created.to_datetime()
                    elif isinstance(created, str):
                        created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    else:
                        continue
                    if created_dt >= cutoff_48h:
                        recent_bookings += 1
                except (ValueError, TypeError, AttributeError):
                    continue

        if recent_bookings == 0 and len(bookings) > 0:
            anomalies.append({
                "type": "no_bookings_48h",
                "description": f"No bookings in the last 48 hours ({len(bookings)} total bookings exist)",
                "severity": "medium",
                "recommendedAction": "Check if booking links are working. Reach out to active artists for feedback.",
            })

        # Check email usage
        email_percent = min(100, round((email_used / 200) * 100))
        if email_percent > 80:
            anomalies.append({
                "type": "email_limit_approaching",
                "description": f"EmailJS usage at {email_percent}% ({email_used}/200)",
                "severity": "high" if email_percent > 95 else "medium",
                "recommendedAction": "Reduce non-critical emails. Plan for next month's quota or upgrade.",
            })

        # Check no signups in 7 days
        recent_signups = 0
        cutoff_7d = now - timedelta(days=7)
        for a in artists:
            created = a.get("createdAt")
            if created:
                try:
                    if hasattr(created, "to_datetime"):
                        created_dt = created.to_datetime()
                    elif isinstance(created, str):
                        created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    else:
                        continue
                    if created_dt >= cutoff_7d:
                        recent_signups += 1
                except (ValueError, TypeError, AttributeError):
                    continue

        if recent_signups == 0 and len(artists) > 0:
            anomalies.append({
                "type": "no_signups_7d",
                "description": "No new artist signups in the last 7 days",
                "severity": "high",
                "recommendedAction": "Increase outreach volume. Review messaging. Try new channels.",
            })

        return anomalies

    # ========== COHORT ANALYSIS ==========
    @staticmethod
    def analyze_cohorts(artists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group artists by signup week, calculate retention per cohort.
        """
        if not artists:
            return []

        # Group by signup week
        cohorts: Dict[str, List[Dict]] = {}
        for a in artists:
            created = a.get("createdAt")
            if not created:
                continue
            try:
                if hasattr(created, "to_datetime"):
                    created_dt = created.to_datetime()
                elif isinstance(created, str):
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                else:
                    continue
                week_key = created_dt.strftime("%Y-W%W")
                if week_key not in cohorts:
                    cohorts[week_key] = []
                cohorts[week_key].append(a)
            except (ValueError, TypeError, AttributeError):
                continue

        results = []
        for week_key in sorted(cohorts.keys()):
            members = cohorts[week_key]
            total = len(members)
            active = sum(1 for a in members if a.get("status") == "active")
            expired = sum(1 for a in members if a.get("status") == "expired")
            trial = sum(1 for a in members if a.get("status") == "trial")
            retention = round((active / total) * 100, 1) if total > 0 else 0

            results.append({
                "cohort": week_key,
                "total": total,
                "active": active,
                "expired": expired,
                "trial": trial,
                "retentionPercent": retention,
            })

        return results

    # ========== FUNNEL ANALYSIS ==========
    @staticmethod
    def analyze_funnel(
        artists: List[Dict[str, Any]],
        outreach_logs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate conversion rates: DM → Reply → Signup → Active → Paid.
        """
        # Count from outreach logs
        dms_sent = sum(1 for o in outreach_logs if o.get("action") == "dm_sent")
        replies = sum(1 for o in outreach_logs if o.get("action") == "reply_received")

        total = len(artists)
        active = sum(1 for a in artists if a.get("status") == "active")
        trial = sum(1 for a in artists if a.get("status") == "trial")
        expired = sum(1 for a in artists if a.get("status") == "expired")

        # Rates
        dm_to_reply = round((replies / dms_sent) * 100, 1) if dms_sent > 0 else 0
        reply_to_signup = round((total / replies) * 100, 1) if replies > 0 else 0
        signup_to_active = round((active / (active + trial)) * 100, 1) if (active + trial) > 0 else 0

        return {
            "stages": {
                "dmsSent": dms_sent,
                "replies": replies,
                "signups": total,
                "active": active,
            },
            "conversionRates": {
                "dmToReply": dm_to_reply,
                "replyToSignup": reply_to_signup,
                "signupToActive": signup_to_active,
            },
        }

    # ========== TREND DETECTION ==========
    @staticmethod
    def detect_trends(daily_snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simple linear regression on metric trajectories using numpy.
        Requires at least 3 data points.
        """
        if len(daily_snapshots) < 3:
            return {"message": "Not enough data for trend detection (need 3+ snapshots)", "trends": {}}

        # Sort by date ascending
        sorted_snapshots = sorted(
            daily_snapshots,
            key=lambda s: s.get("createdAt", ""),
        )

        trends = {}
        metrics_to_track = [
            "totalArtists", "activeArtists", "mrr", "totalBookings",
            "healthScore", "dmsSent",
        ]

        for metric in metrics_to_track:
            values = []
            for snap in sorted_snapshots:
                val = snap.get("metrics", {}).get(metric, 0) if "metrics" in snap else snap.get(metric, 0)
                values.append(val or 0)

            if len(values) < 3:
                continue

            x = np.arange(len(values))
            y = np.array(values, dtype=float)

            try:
                # Linear regression: y = slope * x + intercept
                slope, intercept = np.polyfit(x, y, 1)
                current = values[-1]
                previous = values[-2] if len(values) >= 2 else current

                if slope > 0.5:
                    direction = "up"
                elif slope < -0.5:
                    direction = "down"
                else:
                    direction = "flat"

                change_percent = round(((current - previous) / previous) * 100, 1) if previous != 0 else 0

                trends[metric] = {
                    "direction": direction,
                    "slope": round(float(slope), 3),
                    "current": current,
                    "previous": previous,
                    "changePercent": change_percent,
                }
            except np.linalg.LinAlgError:
                continue

        return {"trends": trends, "dataPoints": len(sorted_snapshots)}

    # ========== CORRELATION DISCOVERY ==========
    @staticmethod
    def discover_correlations(snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find correlations between metric pairs using numpy.
        """
        if len(snapshots) < 5:
            return []

        metrics = ["totalArtists", "activeArtists", "mrr", "totalBookings", "dmsSent"]
        pairs = []

        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                x_vals = []
                y_vals = []
                for snap in snapshots:
                    data = snap.get("metrics", snap)
                    x_vals.append(data.get(metrics[i], 0) or 0)
                    y_vals.append(data.get(metrics[j], 0) or 0)

                if len(x_vals) < 3:
                    continue

                try:
                    corr = np.corrcoef(x_vals, y_vals)[0, 1]
                    if not np.isnan(corr) and abs(corr) > 0.3:
                        pairs.append({
                            "metric1": metrics[i],
                            "metric2": metrics[j],
                            "correlation": round(float(corr), 3),
                            "strength": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.5 else "weak",
                        })
                except (ValueError, np.linalg.LinAlgError):
                    continue

        return sorted(pairs, key=lambda p: abs(p["correlation"]), reverse=True)

    # ========== FULL DASHBOARD ANALYSIS ==========
    def analyze_dashboard(
        self,
        artists: List[Dict[str, Any]],
        bookings: List[Dict[str, Any]],
        outreach_logs: List[Dict[str, Any]],
        payments: List[Dict[str, Any]],
        whatsapp_stats: List[Dict[str, Any]],
        side_hustle: List[Dict[str, Any]],
        honeypot_logs: List[Dict[str, Any]],
        snapshots: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Complete dashboard analysis matching frontend metrics exactly.
        """
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)

        # --- Artists ---
        total = len(artists)
        active = sum(1 for a in artists if a.get("status") == "active")
        trial = sum(1 for a in artists if a.get("status") == "trial")
        expired = sum(1 for a in artists if a.get("status") == "expired")
        standard = sum(
            1 for a in artists
            if (a.get("tier") or "standard") == "standard" and a.get("status") == "active"
        )
        pro = sum(
            1 for a in artists
            if a.get("tier") == "pro" and a.get("status") == "active"
        )
        premium = sum(
            1 for a in artists
            if a.get("tier") == "premium" and a.get("status") == "active"
        )
        mrr = (standard * 19) + (pro * 39) + (premium * 59)

        # New signups in 7 days
        new_signups_7d = 0
        for a in artists:
            created = a.get("createdAt")
            if created:
                try:
                    if hasattr(created, "to_datetime"):
                        dt = created.to_datetime()
                    elif isinstance(created, str):
                        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    else:
                        continue
                    if dt >= week_ago:
                        new_signups_7d += 1
                except (ValueError, TypeError, AttributeError):
                    pass

        # --- Bookings ---
        total_bookings = len(bookings)
        approved_b = sum(1 for b in bookings if b.get("status") == "approved")
        pending_b = sum(1 for b in bookings if not b.get("status") or b.get("status") == "pending")
        declined_b = sum(1 for b in bookings if b.get("status") == "declined")

        bookings_7d = 0
        bookings_30d = 0
        for b in bookings:
            created = b.get("createdAt")
            if created:
                try:
                    if hasattr(created, "to_datetime"):
                        dt = created.to_datetime()
                    elif isinstance(created, str):
                        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    else:
                        continue
                    if dt >= week_ago:
                        bookings_7d += 1
                    if dt >= month_ago:
                        bookings_30d += 1
                except (ValueError, TypeError, AttributeError):
                    pass

        # --- Outreach ---
        outreach_recent = []
        for o in outreach_logs:
            created = o.get("createdAt")
            if created:
                try:
                    if hasattr(created, "to_datetime"):
                        dt = created.to_datetime()
                    elif isinstance(created, str):
                        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    else:
                        continue
                    if dt >= month_ago:
                        outreach_recent.append(o)
                except (ValueError, TypeError, AttributeError):
                    pass

        dms_sent = sum(1 for o in outreach_recent if o.get("action") == "dm_sent")
        replies = sum(1 for o in outreach_recent if o.get("action") == "reply_received")
        reply_rate = round((replies / dms_sent) * 100) if dms_sent > 0 else 0

        # --- Conversion & Churn ---
        conversion_rate = round((active / (trial + active)) * 100) if (trial + active) > 0 else 0
        churn_rate = round((expired / total) * 100) if total > 0 else 0

        # --- Days to 10 ---
        weekly_growth = new_signups_7d
        days_to_10 = (
            max(0, round(((10 - total) / (weekly_growth / 7))))
            if weekly_growth > 0 and total < 10
            else 0 if total >= 10 else 999
        )

        # --- Projected MRR ---
        projected_mrr = ((standard + 5) * 19) + ((pro + 2) * 39) + ((premium + 1) * 59)

        # --- Side Hustle ---
        side_revenue = sum(
            s.get("amount", 0) or 0
            for s in side_hustle
            if s.get("status") == "sold"
        )
        side_completed = sum(1 for s in side_hustle if s.get("status") == "sold")

        # --- WhatsApp ---
        whatsapp_sorted = sorted(
            whatsapp_stats,
            key=lambda w: (
                w.get("createdAt").to_millis()
                if hasattr(w.get("createdAt", None), "to_millis")
                else 0
            ),
            reverse=True,
        )
        w_members = whatsapp_sorted[0].get("members", 0) if whatsapp_sorted else 0
        w_referrals = whatsapp_sorted[0].get("referrals", 0) if whatsapp_sorted else 0

        # --- Honeypot ---
        honeypot_recent = 0
        for h in honeypot_logs:
            ts = h.get("timestamp")
            if ts:
                try:
                    if hasattr(ts, "to_datetime"):
                        dt = ts.to_datetime()
                    elif isinstance(ts, str):
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        continue
                    if dt >= week_ago:
                        honeypot_recent += 1
                except (ValueError, TypeError, AttributeError):
                    pass

        # --- Email usage ---
        email_used = total_bookings + approved_b
        email_percent = min(100, round((email_used / 200) * 100))

        # --- Health score ---
        health_score = (
            min(20, (total / 10) * 20 if total < 10 else 20) +
            min(30, (active / 5) * 30 if active < 5 else 30) +
            min(25, (bookings_30d / 50) * 25 if bookings_30d < 50 else 25) +
            (15 if total > 0 else 0) +
            (10 if total_bookings > 0 else 0)
        )
        health_score = min(100, round(health_score))
        if health_score < 20:
            health_label = "CRITICAL"
        elif health_score < 40:
            health_label = "POOR"
        elif health_score < 60:
            health_label = "FAIR"
        elif health_score < 80:
            health_label = "GOOD"
        else:
            health_label = "EXCELLENT"

        # --- Pending payments ---
        pending_payments = [
            p for p in payments
            if p.get("status") in ("awaiting_payment", "evidence_sent")
        ]
        pending_total = sum(p.get("amount", 0) or 0 for p in pending_payments)

        # --- Anomalies ---
        anomalies = self.detect_anomalies(artists, bookings, email_used)

        # --- Trends ---
        trend_data = self.detect_trends(snapshots)

        # --- Cohorts ---
        cohort_data = self.analyze_cohorts(artists)

        # --- Funnel ---
        funnel_data = self.analyze_funnel(artists, outreach_logs)

        # --- Correlations ---
        correlation_data = self.discover_correlations(snapshots)

        return {
            "totalArtists": total,
            "activeArtists": active,
            "trialArtists": trial,
            "expiredArtists": expired,
            "standardActive": standard,
            "proActive": pro,
            "premiumActive": premium,
            "mrr": mrr,
            "totalBookings": total_bookings,
            "approvedBookings": approved_b,
            "pendingBookings": pending_b,
            "declinedBookings": declined_b,
            "bookings7d": bookings_7d,
            "bookings30d": bookings_30d,
            "dmsSent": dms_sent,
            "repliesReceived": replies,
            "replyRate": reply_rate,
            "newSignups7d": new_signups_7d,
            "conversionRate": conversion_rate,
            "churnRate": churn_rate,
            "daysTo10": days_to_10,
            "projectedMRR": projected_mrr,
            "sideRevenue": side_revenue,
            "sideCompleted": side_completed,
            "whatsappMembers": w_members,
            "whatsappReferrals": w_referrals,
            "honeypotRecent": honeypot_recent,
            "emailUsed": email_used,
            "emailPercent": email_percent,
            "healthScore": health_score,
            "healthLabel": health_label,
            "pendingPaymentsCount": len(pending_payments),
            "pendingPaymentsTotal": pending_total,
            "anomalies": anomalies,
            "trends": trend_data.get("trends", {}),
            "cohorts": cohort_data,
            "funnel": funnel_data,
            "correlations": correlation_data,
            "lastUpdated": now.isoformat(),
        }
```

---

What this does:

· 8 analysis methods, all pure Python + numpy — zero ML libraries
· score_lead() — weights followers, booking method, status, priority
· predict_churn() — checks trial expiry, inactivity, no bookings
· grade_message() — scores A-F on length, personalization, questions, spam triggers
· detect_anomalies() — flags no bookings 48h, email >80%, no signups 7d
· analyze_cohorts() — groups artists by signup week, calculates retention
· analyze_funnel() — DM → Reply → Signup → Active conversion rates
· detect_trends() — numpy linear regression on 6 metrics
· discover_correlations() — numpy correlation between metric pairs
· analyze_dashboard() — full dashboard output matching your frontend exactly (same field names, same health score formula)

---

Approve this and I'll write File #8: engine/ai_enhancer.py
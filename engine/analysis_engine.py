"""
InkFlow Analysis Engine
Cohort analysis, funnel tracking, trend detection, comparative analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json

class AnalysisEngine:
    """Complete data analysis pipeline"""
    
    def __init__(self):
        self.cache = {}
    
    # ========== COHORT ANALYSIS ==========
    
    def cohort_analysis(self, artists: List[Dict], bookings: List[Dict]) -> Dict:
        """Analyze user retention by signup cohort"""
        # Group artists by signup week
        cohorts = defaultdict(list)
        
        for artist in artists:
            created = artist.get('createdAt')
            if created:
                try:
                    if hasattr(created, 'timestamp'):
                        date = datetime.fromtimestamp(created.timestamp())
                    elif hasattr(created, 'toDate'):
                        date = created.toDate()
                    else:
                        date = datetime.fromisoformat(str(created)) if isinstance(created, str) else None
                    
                    if date:
                        week = date.strftime("%Y-W%W")
                        cohorts[week].append(artist)
                except:
                    pass
        
        # Calculate retention for each cohort
        results = []
        for week, cohort_artists in sorted(cohorts.items()):
            total = len(cohort_artists)
            active = len([a for a in cohort_artists if a.get('status') == 'active'])
            trial = len([a for a in cohort_artists if a.get('status') == 'trial'])
            expired = len([a for a in cohort_artists if a.get('status') == 'expired'])
            
            # Bookings from this cohort
            cohort_emails = [a.get('email') for a in cohort_artists if a.get('email')]
            cohort_bookings = len([b for b in bookings if b.get('artistEmail') in cohort_emails])
            
            results.append({
                "cohort": week,
                "total": total,
                "active": active,
                "trial": trial,
                "expired": expired,
                "retention_rate": round(active / max(total, 1) * 100, 1),
                "bookings": cohort_bookings,
                "bookings_per_artist": round(cohort_bookings / max(total, 1), 1)
            })
        
        return {
            "cohorts": results,
            "total_cohorts": len(results),
            "analysis_date": datetime.now().isoformat()
        }
    
    # ========== FUNNEL ANALYSIS ==========
    
    def funnel_analysis(self, artists: List[Dict], outreach: List[Dict], bookings: List[Dict]) -> Dict:
        """Analyze conversion funnel: DM → Reply → Signup → Active → Paid"""
        # Stage 1: Outreach
        dms_sent = len([o for o in outreach if o.get('action') == 'dm_sent'])
        
        # Stage 2: Replies
        replies = len([o for o in outreach if o.get('action') == 'reply_received'])
        
        # Stage 3: Signups
        signups = len(artists)
        
        # Stage 4: Active users
        active = len([a for a in artists if a.get('status') == 'active'])
        
        # Stage 5: Paid users (any tier above standard that's active)
        paid = len([a for a in artists if a.get('status') == 'active' and a.get('tier', 'standard') != 'standard'])
        
        # Calculate conversion rates
        dm_to_reply = round(replies / max(dms_sent, 1) * 100, 1)
        reply_to_signup = round(signups / max(replies, 1) * 100, 1)
        signup_to_active = round(active / max(signups, 1) * 100, 1)
        active_to_paid = round(paid / max(active, 1) * 100, 1)
        
        # Identify biggest drop-off
        stages = [
            {"stage": "DM Sent → Reply", "rate": dm_to_reply, "from": dms_sent, "to": replies},
            {"stage": "Reply → Signup", "rate": reply_to_signup, "from": replies, "to": signups},
            {"stage": "Signup → Active", "rate": signup_to_active, "from": signups, "to": active},
            {"stage": "Active → Paid", "rate": active_to_paid, "from": active, "to": paid}
        ]
        
        biggest_drop = min(stages, key=lambda s: s['rate'])
        
        return {
            "funnel": stages,
            "total_dms": dms_sent,
            "total_replies": replies,
            "total_signups": signups,
            "total_active": active,
            "total_paid": paid,
            "overall_conversion": round(paid / max(dms_sent, 1) * 100, 1),
            "biggest_bottleneck": biggest_drop['stage'],
            "bottleneck_rate": biggest_drop['rate'],
            "analysis_date": datetime.now().isoformat()
        }
    
    # ========== TREND DETECTION ==========
    
    def detect_trends(self, metrics_history: List[Dict]) -> Dict:
        """Detect trends in time-series metrics"""
        if not metrics_history or len(metrics_history) < 3:
            return {"status": "insufficient_data"}
        
        trends = {}
        
        metrics_to_check = ['totalArtists', 'activeArtists', 'mrr', 'totalBookings', 
                           'bookings7d', 'dmsSent', 'repliesReceived', 'healthScore']
        
        for metric in metrics_to_check:
            values = [m.get(metric, 0) for m in metrics_history if m.get(metric) is not None]
            if len(values) < 3:
                continue
            
            # Simple linear regression for trend
            x = np.arange(len(values))
            y = np.array(values)
            
            if np.std(y) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                
                # Calculate trend strength
                correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
                
                # Predict next value
                next_value = slope * len(values) + intercept
                
                trend_direction = "up" if slope > 0.1 else "down" if slope < -0.1 else "stable"
                trend_strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"
                
                trends[metric] = {
                    "direction": trend_direction,
                    "strength": trend_strength,
                    "slope": round(float(slope), 3),
                    "correlation": round(float(correlation), 3),
                    "current": float(values[-1]) if values else 0,
                    "predicted_next": round(float(max(next_value, 0)), 1),
                    "change_percent": round((values[-1] - values[0]) / max(values[0], 1) * 100, 1) if values[0] > 0 else 0
                }
        
        return {
            "trends": trends,
            "data_points": len(metrics_history),
            "analysis_date": datetime.now().isoformat()
        }
    
    # ========== COMPARATIVE ANALYSIS ==========
    
    def compare_periods(self, current_metrics: Dict, previous_metrics: Dict) -> Dict:
        """Compare two time periods"""
        comparisons = {}
        
        metrics_to_compare = [
            'totalArtists', 'activeArtists', 'mrr', 'totalBookings', 
            'bookings7d', 'bookings30d', 'dmsSent', 'repliesReceived', 'healthScore'
        ]
        
        for metric in metrics_to_compare:
            current = current_metrics.get(metric, 0) or 0
            previous = previous_metrics.get(metric, 0) or 0
            
            change = current - previous
            if previous > 0:
                change_percent = round((change / previous) * 100, 1)
            else:
                change_percent = 100 if current > 0 else 0
            
            comparisons[metric] = {
                "current": current,
                "previous": previous,
                "change": change,
                "change_percent": change_percent,
                "direction": "up" if change > 0 else "down" if change < 0 else "unchanged"
            }
        
        # Overall assessment
        improving = len([c for c in comparisons.values() if c['direction'] == 'up'])
        declining = len([c for c in comparisons.values() if c['direction'] == 'down'])
        
        return {
            "comparisons": comparisons,
            "metrics_improving": improving,
            "metrics_declining": declining,
            "overall_assessment": "improving" if improving > declining else "declining" if declining > improving else "stable"
        }
    
    # ========== CORRELATION DISCOVERY ==========
    
    def discover_correlations(self, metrics_history: List[Dict]) -> Dict:
        """Discover correlations between different metrics"""
        if len(metrics_history) < 5:
            return {"status": "insufficient_data"}
        
        correlations = []
        metrics_to_check = ['dmsSent', 'repliesReceived', 'totalBookings', 'activeArtists', 'mrr']
        
        for i, metric1 in enumerate(metrics_to_check):
            for metric2 in metrics_to_check[i+1:]:
                values1 = [m.get(metric1, 0) for m in metrics_history if m.get(metric1) is not None and m.get(metric2) is not None]
                values2 = [m.get(metric2, 0) for m in metrics_history if m.get(metric1) is not None and m.get(metric2) is not None]
                
                if len(values1) >= 3:
                    corr = np.corrcoef(values1, values2)[0, 1]
                    if abs(corr) > 0.5:
                        correlations.append({
                            "metric1": metric1,
                            "metric2": metric2,
                            "correlation": round(float(corr), 3),
                            "strength": "strong" if abs(corr) > 0.7 else "moderate",
                            "direction": "positive" if corr > 0 else "negative",
                            "insight": self._generate_correlation_insight(metric1, metric2, corr)
                        })
        
        correlations.sort(key=lambda c: abs(c['correlation']), reverse=True)
        
        return {
            "correlations": correlations[:10],
            "total_discovered": len(correlations),
            "analysis_date": datetime.now().isoformat()
        }
    
    def _generate_correlation_insight(self, metric1: str, metric2: str, correlation: float) -> str:
        """Generate human-readable insight from correlation"""
        insights = {
            ('dmsSent', 'repliesReceived'): "More outreach DMs lead to more replies — consistent effort pays off.",
            ('dmsSent', 'totalBookings'): "Outreach volume correlates with booking activity — keep sending DMs.",
            ('activeArtists', 'mrr'): "Active artists directly drive revenue — activation is key.",
            ('repliesReceived', 'totalBookings'): "Artists who engage with outreach tend to book more.",
        }
        
        key = tuple(sorted([metric1, metric2]))
        if key in insights:
            return insights[key]
        
        return f"{'Positive' if correlation > 0 else 'Negative'} correlation between {metric1} and {metric2}."
    
    # ========== FULL DASHBOARD ANALYSIS ==========
    
    def full_analysis(self, artists: List[Dict], bookings: List[Dict], 
                     outreach: List[Dict], metrics_history: List[Dict]) -> Dict:
        """Run complete analysis suite"""
        return {
            "cohorts": self.cohort_analysis(artists, bookings),
            "funnel": self.funnel_analysis(artists, outreach, bookings),
            "trends": self.detect_trends(metrics_history) if metrics_history else {"status": "no_history"},
            "correlations": self.discover_correlations(metrics_history) if metrics_history else {"status": "no_history"},
            "analysis_timestamp": datetime.now().isoformat()
        }


# Singleton instance
analysis_engine = AnalysisEngine()

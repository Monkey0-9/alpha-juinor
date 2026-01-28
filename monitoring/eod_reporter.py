"""
End-of-Day Reporter.
Aggregates cycle statistics and generates daily summary JSON.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class EODReporter:
    """Generates end-of-day summary reports"""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(
        self,
        date: str,
        cycles_run: int,
        all_decisions: List[Any],
        provider_usage: Dict[str, int],
        quality_stats: Dict[str, Any],
        bandit_stats: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate EOD summary report.

        Args:
            date: Date string (YYYY-MM-DD)
            cycles_run: Number of cycles executed
            all_decisions: List of Decision objects
            provider_usage: Dict of provider usage counts
            quality_stats: Data quality statistics
            bandit_stats: Provider bandit statistics (optional)

        Returns:
            Report dict
        """
        # Decision breakdown
        decision_counts = Counter(d.final_decision.value for d in all_decisions)

        # Top reject reasons
        reject_reasons = []
        for d in all_decisions:
            if d.final_decision.value == "REJECT":
                reject_reasons.extend(d.reason_codes)
        top_reject_reasons = Counter(reject_reasons).most_common(10)

        # Conviction stats
        convictions = [d.conviction for d in all_decisions if d.conviction > 0]
        avg_conviction = sum(convictions) / len(convictions) if convictions else 0.0

        # Quality stats
        avg_quality = (quality_stats.get("quality_pass", 0) /
                      quality_stats.get("total_fetches", 1))

        report = {
            "date": date,
            "cycles_run": cycles_run,
            "total_decisions": len(all_decisions),
            "decision_breakdown": dict(decision_counts),
            "provider_usage": provider_usage,
            "top_reject_reasons": [
                {"reason": reason, "count": count}
                for reason, count in top_reject_reasons
            ],
            "performance": {
                "avg_conviction": round(avg_conviction, 4),
                "avg_quality_score": round(avg_quality, 4),
                "quality_pass_rate": round(avg_quality, 4)
            },
            "bandit_stats": bandit_stats or {}
        }

        return report

    def write_report(self, report: Dict[str, Any], date: str = None):
        """Write report to JSON file"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        filename = f"eod_summary_{date}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"EOD report written to {filepath}")
        return filepath

    def generate_and_write(
        self,
        cycles_run: int,
        all_decisions: List[Any],
        provider_usage: Dict[str, int],
        quality_stats: Dict[str, Any],
        bandit_stats: Dict[str, Any] = None,
        date: str = None
    ) -> str:
        """Generate and write report in one call"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        report = self.generate_report(
            date, cycles_run, all_decisions,
            provider_usage, quality_stats, bandit_stats
        )

        return self.write_report(report, date)

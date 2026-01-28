"""
PM-Style Cycle Summary for Terminal Output.
Provides structured, readable summary of cycle execution.
"""

import logging
from typing import List, Dict, Any
from collections import Counter
from contracts import DecisionRecord
# from contracts import Decision, decision_enum # Removed

logger = logging.getLogger(__name__)


def print_cycle_summary(
    cycle_id: str,
    results: List[DecisionRecord],
    universe_size: int,
    duration_sec: float,
    provider_stats: Dict[str, int],
    quality_stats: Dict[str, Any],
    bandit_stats: Dict[str, Any] = None
):
    """
    Print PM-style terminal summary for a cycle.

    Args:
        cycle_id: Unique cycle identifier
        results: List of DecisionRecord objects
        universe_size: Expected universe size
        duration_sec: Cycle duration in seconds
        provider_stats: Provider usage counts
        quality_stats: Data quality statistics
        bandit_stats: Provider bandit statistics (optional)
    """

    # Decision breakdown
    execute = sum(1 for d in results if d.final_decision == "EXECUTE")
    hold = sum(1 for d in results if d.final_decision == "HOLD")
    reject = sum(1 for d in results if d.final_decision == "REJECT")
    error = sum(1 for d in results if d.final_decision == "ERROR")

    total = len(results)
    coverage_pct = (total / universe_size * 100) if universe_size > 0 else 0

    # Top reject reasons
    reject_reasons = Counter()
    for d in results:
        if d.final_decision == "REJECT":
            reject_reasons.update(d.reason_codes)

    # Quality metrics
    quality_pass = quality_stats.get("quality_pass", 0)
    quality_total = quality_stats.get("total_fetches", 1)
    quality_pct = (quality_pass / quality_total * 100) if quality_total > 0 else 0

    # Provider breakdown
    provider_list = ', '.join(f"{k}:{v}" for k, v in sorted(provider_stats.items(), key=lambda x: -x[1])[:3])

    # Coverage status
    coverage_status = "✓" if total == universe_size else "✗"

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          CYCLE SUMMARY                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Cycle ID:        {cycle_id[:40]:<40}                        ║
║ Duration:        {duration_sec:>6.1f}s                                                   ║
║ Universe:        {universe_size:>4} symbols configured                                   ║
║ Coverage:        {coverage_status} {total:>4}/{universe_size:<4} ({coverage_pct:>5.1f}%)                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ DECISIONS                                                                     ║
║   EXECUTE:       {execute:>4} ({execute/total*100 if total > 0 else 0:>5.1f}%)                                          ║
║   HOLD:          {hold:>4} ({hold/total*100 if total > 0 else 0:>5.1f}%)                                          ║
║   REJECT:        {reject:>4} ({reject/total*100 if total > 0 else 0:>5.1f}%)                                          ║
║   ERROR:         {error:>4} ({error/total*100 if total > 0 else 0:>5.1f}%)                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ DATA PROVIDERS                                                                ║
║   {provider_list:<74} ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ DATA QUALITY                                                                  ║
║   Pass Rate:     {quality_pct:>5.1f}% ({quality_pass}/{quality_total})                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ TOP REJECT REASONS                                                            ║
║   1. {reject_reasons.most_common(1)[0][0] if reject_reasons else 'N/A':<50} {reject_reasons.most_common(1)[0][1] if reject_reasons else 0:>4}           ║
║   2. {reject_reasons.most_common(2)[1][0] if len(reject_reasons) > 1 else 'N/A':<50} {reject_reasons.most_common(2)[1][1] if len(reject_reasons) > 1 else 0:>4}           ║
║   3. {reject_reasons.most_common(3)[2][0] if len(reject_reasons) > 2 else 'N/A':<50} {reject_reasons.most_common(3)[2][1] if len(reject_reasons) > 2 else 0:>4}           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ AUDIT: ✓ {total} records written to runtime/audit.db                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Log structured JSON for programmatic parsing
    summary_json = {
        "cycle_id": cycle_id,
        "duration_sec": duration_sec,
        "universe_size": universe_size,
        "decisions_generated": total,
        "coverage_pct": coverage_pct,
        "decisions": {
            "EXECUTE": execute,
            "HOLD": hold,
            "REJECT": reject,
            "ERROR": error
        },
        "providers": provider_stats,
        "quality_pass_rate": quality_pct,
        "top_reject_reasons": [
            {"reason": reason, "count": count}
            for reason, count in reject_reasons.most_common(5)
        ]
    }

    logger.info(f"CYCLE_SUMMARY_JSON: {summary_json}")

    # Warnings
    if total != universe_size:
        logger.error(f"⚠️  COVERAGE MISMATCH: Expected {universe_size}, got {total}")

    if error > 0:
        logger.warning(f"⚠️  {error} ERROR decisions - review audit log")

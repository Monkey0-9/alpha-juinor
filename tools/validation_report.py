"""
Data Quality Validation and Reporting.

Generates comprehensive validation reports for backfilled data.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

import pandas as pd

from database.manager import get_db

logger = logging.getLogger(__name__)


class DataQualityReporter:
    """
    Generates validation reports for market data quality.
    """

    def __init__(self, db=None):
        self.db = db or get_db()

    def generate_validation_report(self, start_date: str, end_date: str,
                                   output_path: str = None) -> Dict:
        """
        Generate comprehensive validation report.

        Args:
            start_date: Report start date
            end_date: Report end date
            output_path: Optional path to save JSON report

        Returns:
            Report dictionary
        """
        report = {
            'report_id': f"validation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.utcnow().isoformat(),
            'period': {'start': start_date, 'end': end_date},
            'summary': self._generate_summary(start_date, end_date),
            'symbol_details': self._generate_symbol_details(start_date, end_date),
            'provider_stats': self._generate_provider_stats(),
            'issues_summary': self._generate_issues_summary(start_date, end_date),
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Validation report saved to {output_path}")

        return report

    def _generate_summary(self, start_date: str, end_date: str) -> Dict:
        """Generate summary statistics"""
        quality_summary = self.db.get_quality_summary()
        coverage = self.db.get_symbol_coverage(start_date, end_date)

        return {
            'total_symbols': coverage['symbols_with_data'],
            'avg_days_per_symbol': round(coverage['average_days_per_symbol'], 1),
            'overall_quality_score': round(quality_summary['avg_quality_score'], 4),
            'min_quality_score': round(quality_summary['min_quality_score'], 4),
            'low_quality_symbols': quality_summary['low_quality_count'],
            'critical_symbols': quality_summary['critical_count'],
            'expected_trading_days': self._get_expected_trading_days(start_date, end_date),
        }

    def _generate_symbol_details(self, start_date: str, end_date: str) -> List[Dict]:
        """Generate per-symbol details"""
        conn = self.db.get_connection()

        cursor = conn.execute('''
            SELECT
                d.symbol,
                d.quality_score,
                d.row_count,
                d.missing_pct,
                d.issues_json,
                d.provider,
                MIN(h.date) as first_date,
                MAX(h.date) as last_date
            FROM data_quality_log d
            LEFT JOIN price_history_daily h ON d.symbol = h.symbol
            WHERE d.date >= ? AND d.date <= ?
            GROUP BY d.symbol
            ORDER BY d.quality_score ASC
        ''', (start_date, end_date))

        details = []
        for row in cursor.fetchall():
            issues = json.loads(row['issues_json']) if row['issues_json'] else []
            details.append({
                'symbol': row['symbol'],
                'quality_score': row['quality_score'],
                'row_count': row['row_count'],
                'missing_pct': row['missing_pct'],
                'issues': issues,
                'provider': row['provider'],
                'date_range': {
                    'first': row['first_date'],
                    'last': row['last_date']
                },
                'status': self._get_status(row['quality_score'])
            })

        return details

    def _generate_provider_stats(self) -> Dict:
        """Generate provider statistics"""
        provider_metrics = self.db.get_provider_metrics(days=90)

        stats = {}
        for row in provider_metrics:
            name = row['provider_name']
            if name not in stats:
                stats[name] = {
                    'total_pulls': 0,
                    'total_successes': 0,
                    'avg_latency_ms': 0,
                    'avg_quality_score': 0,
                    'success_rate': 0
                }

            stats[name]['total_pulls'] += row['pulls']
            stats[name]['total_successes'] += row['successes']

        # Calculate success rates
        for name, s in stats.items():
            if s['total_pulls'] > 0:
                s['success_rate'] = round(s['total_successes'] / s['total_pulls'], 4)

        return stats

    def _generate_issues_summary(self, start_date: str, end_date: str) -> Dict:
        """Generate summary of issues found"""
        conn = self.db.get_connection()

        cursor = conn.execute('''
            SELECT issues_json FROM data_quality_log
            WHERE date >= ? AND date <= ? AND issues_json IS NOT NULL
        ''', (start_date, end_date))

        issue_counts = {}
        for row in cursor.fetchall():
            issues = json.loads(row['issues_json'])
            for issue in issues:
                issue_type = issue.split('(')[0] if '(' in issue else issue
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        return {
            'total_issues': sum(issue_counts.values()),
            'by_type': issue_counts
        }

    def _get_status(self, quality_score: float) -> str:
        """Get status based on quality score"""
        if quality_score >= 0.8:
            return 'excellent'
        elif quality_score >= 0.6:
            return 'good'
        elif quality_score >= 0.4:
            return 'marginal'
        else:
            return 'critical'

    def _get_expected_trading_days(self, start_date: str, end_date: str) -> int:
        """Get expected trading days in range"""
        all_days = pd.date_range(start=start_date, end=end_date, freq='B')
        return len(all_days)

    def print_summary(self, report: Dict):
        """Print human-readable summary"""
        print("\n" + "=" * 80)
        print("DATA QUALITY VALIDATION REPORT")
        print("=" * 80)

        summary = report['summary']
        print(f"\nPeriod: {summary['period']['start']} to {summary['period']['end']}")
        print(f"Symbols with Data: {summary['total_symbols']}")
        print(f"Avg Days/Symbol: {summary['avg_days_per_symbol']}")
        print(f"Overall Quality Score: {summary['overall_quality_score']:.4f}")

        print(f"\nQuality Distribution:")
        print(f"  Excellent (≥0.8): {len([s for s in report['symbol_details'] if s['quality_score'] >= 0.8])}")
        print(f"  Good (0.6-0.8): {len([s for s in report['symbol_details'] if 0.6 <= s['quality_score'] < 0.8])}")
        print(f"  Marginal (0.4-0.6): {len([s for s in report['symbol_details'] if 0.4 <= s['quality_score'] < 0.4])}")
        print(f"  Critical (<0.4): {len([s for s in report['symbol_details'] if s['quality_score'] < 0.4])}")

        print(f"\nTop Issues:")
        for issue_type, count in sorted(report['issues_summary']['by_type'].items(),
                                         key=lambda x: -x[1])[:5]:
            print(f"  {issue_type}: {count}")

        print("\n" + "=" * 80)


def generate_report(start_date: str, end_date: str, output: str = None):
    """Generate validation report"""
    reporter = DataQualityReporter()
    report = reporter.generate_validation_report(start_date, end_date, output)
    reporter.print_summary(report)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Data Quality Report')
    parser.add_argument('--start', type=str, default='2021-01-19')
    parser.add_argument('--end', type=str, default='2026-01-19')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')

    args = parser.parse_args()

    report = generate_report(args.start, args.end, args.output)
    print(json.dumps(report, indent=2, default=str))


#!/usr/bin/env python3
"""
REGULATORY REPORTING ENGINE
===========================

Automated compliance and regulatory reporting for SEC, FINRA, CFTC.

Features:
- Form PF (Private Fund)
- Form ADV (Investment Advisor)
- MiFID II Transaction Reporting
- CAT (Consolidated Audit Trail)
- Real-time surveillance alerts
- Trade reconstruction

Author: MiniQuantFund Compliance Team
"""

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import xml.etree.ElementTree as ET
import csv
import io

import pandas as pd
import numpy as np

from mini_quant_fund.database.timescaledb_cluster import get_timescale_cluster

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of regulatory reports."""
    FORM_PF = "form_pf"
    FORM_ADV = "form_adv"
    MIFID_RTE = "mifid_rte"  # Real-time execution
    MIFID_TSR = "mifid_tsr"  # Transaction summary
    CAT = "cat"  # Consolidated Audit Trail
    LARGE_TRADER = "large_trader"
    SHORT_INTEREST = "short_interest"
    REG_SHO = "reg_sho"


class ReportFrequency(Enum):
    """Report filing frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    EVENT = "event"  # As-needed


@dataclass
class TradeReport:
    """Standardized trade report format."""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'BUY', 'SELL'
    quantity: int
    price: float
    currency: str = "USD"
    exchange: str = ""
    strategy: str = ""
    account: str = ""
    counterparty: str = ""
    execution_venue: str = ""
    order_id: str = ""
    parent_order_id: Optional[str] = None
    
    # Compliance fields
    is_short_sale: bool = False
    is_locate_required: bool = False
    locate_source: Optional[str] = None
    
    # MiFID II fields
    waiver_indicator: Optional[str] = None  # RPRIS, SIZE, ILQD, etc.
    algo_indicator: bool = False
    participant_id: Optional[str] = None
    
    # CAT fields
    imid: str = ""  # Identifying Member ID
    cat_reportable: bool = True


@dataclass
class PositionReport:
    """Position snapshot for regulatory reporting."""
    date: date
    symbol: str
    quantity: int
    market_value: float
    currency: str
    account: str
    is_short: bool = False
    is_borrowed: bool = False


@dataclass
class FormPFData:
    """Form PF required data."""
    reporting_period: str
    fund_name: str
    fund_id: str
    nav: float
    gmv: float  # Gross Market Value
    borrowing: float
    derivatives_exposure: float
    var: float
    stress_test_results: Dict[str, float]
    liquidity_profile: Dict[str, float]
    counterparty_exposure: Dict[str, float]
    risk_factors: List[str]


class RegulatoryReportGenerator:
    """Generate regulatory reports in required formats."""
    
    def __init__(self, fund_name: str = "MiniQuantFund", 
                 fund_id: str = "MQF001"):
        self.fund_name = fund_name
        self.fund_id = fund_id
        self.cluster = get_timescale_cluster()
        
        # Output directory
        self.output_dir = Path("reports/regulatory")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # FORM PF GENERATION
    # =========================================================================
    
    def generate_form_pf(self, as_of_date: Optional[date] = None) -> str:
        """
        Generate Form PF (Private Fund Report).
        
        Form PF is filed quarterly by private fund advisors with >$150M AUM.
        """
        as_of_date = as_of_date or date.today()
        
        # Collect required data
        data = self._collect_form_pf_data(as_of_date)
        
        # Generate XML
        xml_content = self._form_pf_to_xml(data)
        
        # Save
        filename = f"FormPF_{self.fund_id}_{as_of_date.strftime('%Y%m%d')}.xml"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(xml_content)
        
        logger.info(f"Generated Form PF: {filepath}")
        return str(filepath)
    
    def _collect_form_pf_data(self, as_of_date: date) -> FormPFData:
        """Collect data required for Form PF."""
        # Query portfolio data
        query = f"""
            SELECT 
                SUM(nav) as nav,
                SUM(exposure) as gmv,
                SUM(margin_used) as borrowing
            FROM portfolio_snapshots
            WHERE DATE(timestamp) = '{as_of_date}'
            ORDER BY timestamp DESC
            LIMIT 1;
        """
        
        portfolio = self.cluster.query_to_dataframe(query)
        
        # Query risk metrics
        risk_query = f"""
            SELECT 
                MAX(volatility) as var_estimate
            FROM portfolio_snapshots
            WHERE timestamp >= '{as_of_date - timedelta(days=30)}';
        """
        
        risk = self.cluster.query_to_dataframe(risk_query)
        
        return FormPFData(
            reporting_period=as_of_date.strftime("%Y-%m"),
            fund_name=self.fund_name,
            fund_id=self.fund_id,
            nav=portfolio['nav'].iloc[0] if not portfolio.empty else 0,
            gmv=portfolio['gmv'].iloc[0] if not portfolio.empty else 0,
            borrowing=portfolio['borrowing'].iloc[0] if not portfolio.empty else 0,
            derivatives_exposure=0.0,  # Would query derivatives table
            var=risk['var_estimate'].iloc[0] * 1.645 if not risk.empty else 0,  # 95% VaR
            stress_test_results={},
            liquidity_profile={},
            counterparty_exposure={},
            risk_factors=["market", "liquidity", "operational"]
        )
    
    def _form_pf_to_xml(self, data: FormPFData) -> str:
        """Convert Form PF data to SEC XML format."""
        root = ET.Element("FormPF", {
            "version": "1.05",
            "xmlns": "http://www.sec.gov/FormPF"
        })
        
        # Header
        header = ET.SubElement(root, "header")
        ET.SubElement(header, "reportingPeriod").text = data.reporting_period
        ET.SubElement(header, "fundName").text = data.fund_name
        ET.SubElement(header, "fundId").text = data.fund_id
        
        # Section 1: Fund Information
        section1 = ET.SubElement(root, "section1")
        ET.SubElement(section1, "nav").text = str(data.nav)
        ET.SubElement(section1, "grossAssetValue").text = str(data.gmv)
        ET.SubElement(section1, "borrowing").text = str(data.borrowing)
        
        # Section 2: Risk Metrics
        section2 = ET.SubElement(root, "section2")
        ET.SubElement(section2, "valueAtRisk").text = str(data.var)
        
        # Section 3: Liquidity Profile
        section3 = ET.SubElement(root, "section3")
        for bucket, amount in data.liquidity_profile.items():
            liquidity = ET.SubElement(section3, "liquidityBucket")
            ET.SubElement(liquidity, "days").text = bucket
            ET.SubElement(liquidity, "amount").text = str(amount)
        
        # Convert to string
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding='unicode')
    
    # =========================================================================
    # CAT REPORTING
    # =========================================================================
    
    def generate_cat_report(self, trade_date: Optional[date] = None) -> str:
        """
        Generate CAT (Consolidated Audit Trail) report.
        
        CAT requires detailed reporting of all order and trade events.
        """
        trade_date = trade_date or date.today()
        
        # Query all trades for the date
        query = f"""
            SELECT *
            FROM trades
            WHERE DATE(timestamp) = '{trade_date}'
            ORDER BY timestamp;
        """
        
        trades_df = self.cluster.query_to_dataframe(query)
        
        if trades_df.empty:
            logger.warning(f"No trades found for {trade_date}")
            return ""
        
        # Convert to CAT format
        cat_records = []
        for _, trade in trades_df.iterrows():
            record = self._trade_to_cat_record(trade)
            cat_records.append(record)
        
        # Generate pipe-delimited file
        filename = f"CAT_{self.fund_id}_{trade_date.strftime('%Y%m%d')}.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            
            # Header
            writer.writerow([
                'IMID', 'CATReporterIMID', 'EventTimestamp', 'EventType',
                'Symbol', 'EventQuantity', 'EventPrice', 'OrderID', 'Side',
                'ExecutionID', 'TradingSession', 'Exchange', 'Capacity'
            ])
            
            # Records
            for record in cat_records:
                writer.writerow([
                    record.imid,
                    record.imid,  # Self-reported
                    record.timestamp.isoformat(),
                    'TRADE',
                    record.symbol,
                    record.quantity,
                    record.price,
                    record.order_id,
                    record.side,
                    record.trade_id,
                    'REGULAR',
                    record.exchange,
                    'A'  # Agency
                ])
        
        logger.info(f"Generated CAT report: {filepath}")
        return str(filepath)
    
    def _trade_to_cat_record(self, trade: pd.Series) -> TradeReport:
        """Convert database trade to CAT report format."""
        return TradeReport(
            trade_id=str(trade.get('trade_id', '')),
            timestamp=trade.get('timestamp', datetime.utcnow()),
            symbol=str(trade.get('symbol_id', '')),
            side='BUY' if trade.get('side', 0) == 0 else 'SELL',
            quantity=int(trade.get('quantity', 0)),
            price=float(trade.get('price', 0)),
            order_id=str(trade.get('trade_id', '')),
            imid="MQF",
            cat_reportable=True
        )
    
    # =========================================================================
    # MIFID II REPORTING
    # =========================================================================
    
    def generate_mifid_rte(self, report_date: Optional[date] = None) -> str:
        """
        Generate MiFID II Real-Time Execution report.
        
        Required for transactions executed on EU trading venues.
        """
        report_date = report_date or date.today()
        
        query = f"""
            SELECT 
                timestamp,
                symbol_id,
                side,
                price,
                quantity,
                strategy_id,
                exchange_id
            FROM trades
            WHERE DATE(timestamp) = '{report_date}'
            AND exchange_id IN (SELECT id FROM exchanges WHERE jurisdiction = 'EU')
            ORDER BY timestamp;
        """
        
        trades = self.cluster.query_to_dataframe(query)
        
        if trades.empty:
            return ""
        
        # Generate CSV in ARM format
        filename = f"MiFID_RTE_{report_date.strftime('%Y%m%d')}.csv"
        filepath = self.output_dir / filename
        
        # MiFID II ARM format
        trades.to_csv(filepath, index=False)
        
        logger.info(f"Generated MiFID RTE: {filepath}")
        return str(filepath)
    
    # =========================================================================
    # SHORT INTEREST REPORTING
    # =========================================================================
    
    def generate_short_interest_report(self, as_of_date: Optional[date] = None) -> str:
        """
        Generate short interest report for REG SHO compliance.
        
        Report short positions > reporting threshold (typically 0.5% of shares).
        """
        as_of_date = as_of_date or date.today()
        
        query = f"""
            SELECT 
                symbol_id,
                SUM(CASE WHEN side = 1 AND is_short = true THEN quantity ELSE 0 END) as short_quantity,
                SUM(CASE WHEN side = 0 THEN quantity ELSE 0 END) as long_quantity
            FROM trades
            WHERE DATE(timestamp) = '{as_of_date}'
            GROUP BY symbol_id
            HAVING SUM(CASE WHEN side = 1 AND is_short = true THEN quantity ELSE 0 END) > 0;
        """
        
        positions = self.cluster.query_to_dataframe(query)
        
        if positions.empty:
            return ""
        
        filename = f"ShortInterest_{as_of_date.strftime('%Y%m%d')}.csv"
        filepath = self.output_dir / filename
        
        positions.to_csv(filepath, index=False)
        
        logger.info(f"Generated Short Interest report: {filepath}")
        return str(filepath)
    
    # =========================================================================
    # LARGE TRADER REPORTING
    # =========================================================================
    
    def check_large_trader_threshold(self, 
                                     trades_df: pd.DataFrame,
                                     threshold_shares: int = 2000000) -> List[Dict]:
        """
        Check if trades exceed large trader reporting threshold.
        
        Large traders must file Form 13H with SEC.
        """
        # Aggregate daily volume by symbol
        daily_volume = trades_df.groupby('symbol_id')['quantity'].sum()
        
        alerts = []
        for symbol, volume in daily_volume.items():
            if volume >= threshold_shares:
                alerts.append({
                    'symbol': symbol,
                    'volume': volume,
                    'threshold': threshold_shares,
                    'date': date.today().isoformat(),
                    'alert_type': 'LARGE_TRADER_THRESHOLD'
                })
        
        if alerts:
            logger.warning(f"Large trader threshold exceeded: {alerts}")
        
        return alerts
    
    # =========================================================================
    # REPORT SCHEDULING
    # =========================================================================
    
    def generate_all_reports(self, as_of_date: Optional[date] = None) -> Dict[str, str]:
        """Generate all regulatory reports for a date."""
        as_of_date = as_of_date or date.today()
        
        reports = {}
        
        # Quarterly reports
        if as_of_date.month in [3, 6, 9, 12] and as_of_date.day == 31:
            reports['form_pf'] = self.generate_form_pf(as_of_date)
        
        # Daily reports
        reports['cat'] = self.generate_cat_report(as_of_date)
        reports['short_interest'] = self.generate_short_interest_report(as_of_date)
        
        # Conditional reports
        reports['mifid_rte'] = self.generate_mifid_rte(as_of_date)
        
        return reports
    
    def validate_report(self, report_type: ReportType, filepath: str) -> bool:
        """Validate generated report against schema."""
        # Would implement schema validation
        # For now, check file exists and is not empty
        path = Path(filepath)
        return path.exists() and path.stat().st_size > 0
    
    def submit_report(self, report_type: ReportType, filepath: str) -> bool:
        """
        Submit report to regulatory authority.
        
        Would integrate with:
        - SEC EDGAR (Form PF)
        - FINRA CAT
        - ARM for MiFID II
        """
        logger.info(f"Submitting {report_type.value}: {filepath}")
        
        # Placeholder - would implement actual submission
        # via SFTP, API, or web portal
        
        return True


class ComplianceMonitor:
    """Real-time compliance monitoring and alerting."""
    
    def __init__(self):
        self.cluster = get_timescale_cluster()
        self.thresholds = {
            'concentration_limit': 0.1,  # 10% in single position
            'leverage_limit': 2.0,  # 2:1 gross leverage
            'var_limit': 0.02,  # 2% daily VaR
        }
    
    def check_position_limits(self) -> List[Dict]:
        """Check for position limit violations."""
        query = """
            SELECT 
                symbol_id,
                SUM(quantity * CASE WHEN side = 0 THEN 1 ELSE -1 END) as net_position,
                ABS(SUM(quantity * CASE WHEN side = 0 THEN 1 ELSE -1 END)) * price as position_value
            FROM trades
            WHERE timestamp >= NOW() - INTERVAL '1 day'
            GROUP BY symbol_id, price;
        """
        
        positions = self.cluster.query_to_dataframe(query)
        
        if positions.empty:
            return []
        
        # Get total portfolio value
        nav_query = """
            SELECT nav FROM portfolio_snapshots
            ORDER BY timestamp DESC LIMIT 1;
        """
        nav = self.cluster.query_to_dataframe(nav_query)
        total_nav = nav['nav'].iloc[0] if not nav.empty else 1
        
        violations = []
        for _, pos in positions.iterrows():
            concentration = pos['position_value'] / total_nav
            
            if concentration > self.thresholds['concentration_limit']:
                violations.append({
                    'type': 'CONCENTRATION_LIMIT',
                    'symbol': pos['symbol_id'],
                    'concentration': concentration,
                    'threshold': self.thresholds['concentration_limit'],
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        return violations
    
    def check_leverage(self) -> Optional[Dict]:
        """Check leverage limits."""
        query = """
            SELECT 
                exposure,
                nav
            FROM portfolio_snapshots
            ORDER BY timestamp DESC LIMIT 1;
        """
        
        snapshot = self.cluster.query_to_dataframe(query)
        
        if snapshot.empty:
            return None
        
        leverage = snapshot['exposure'].iloc[0] / snapshot['nav'].iloc[0]
        
        if leverage > self.thresholds['leverage_limit']:
            return {
                'type': 'LEVERAGE_LIMIT',
                'current_leverage': leverage,
                'threshold': self.thresholds['leverage_limit'],
                'timestamp': datetime.utcnow().isoformat()
            }
        
        return None
    
    def run_compliance_checks(self) -> Dict:
        """Run all compliance checks."""
        violations = []
        
        # Check position limits
        violations.extend(self.check_position_limits())
        
        # Check leverage
        leverage_violation = self.check_leverage()
        if leverage_violation:
            violations.append(leverage_violation)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'violations': violations,
            'violation_count': len(violations),
            'compliant': len(violations) == 0
        }


# Global instances
_report_generator: Optional[RegulatoryReportGenerator] = None
_compliance_monitor: Optional[ComplianceMonitor] = None


def get_report_generator() -> RegulatoryReportGenerator:
    """Get global report generator."""
    global _report_generator
    if _report_generator is None:
        _report_generator = RegulatoryReportGenerator()
    return _report_generator


def get_compliance_monitor() -> ComplianceMonitor:
    """Get global compliance monitor."""
    global _compliance_monitor
    if _compliance_monitor is None:
        _compliance_monitor = ComplianceMonitor()
    return _compliance_monitor


if __name__ == "__main__":
    # Test regulatory reporting
    print("Testing Regulatory Reporting Engine...")
    
    generator = RegulatoryReportGenerator()
    
    # Generate reports
    reports = generator.generate_all_reports()
    
    print(f"\nGenerated reports:")
    for report_type, filepath in reports.items():
        if filepath:
            print(f"  {report_type}: {filepath}")
    
    # Run compliance checks
    monitor = ComplianceMonitor()
    compliance = monitor.run_compliance_checks()
    
    print(f"\nCompliance Status:")
    print(f"  Compliant: {compliance['compliant']}")
    print(f"  Violations: {compliance['violation_count']}")
    
    if compliance['violations']:
        for v in compliance['violations']:
            print(f"    - {v['type']}: {v}")

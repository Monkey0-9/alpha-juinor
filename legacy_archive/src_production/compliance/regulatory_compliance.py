"""
Regulatory Compliance - Production Implementation
Handles FINRA, SEC, MiFID II compliance requirements
"""

import asyncio
import logging
import json
import hashlib
import hmac
import base64
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import sqlite3
import uuid

logger = logging.getLogger(__name__)

class RegulationType(Enum):
    """Regulation types"""
    FINRA = "finra"
    SEC = "sec"
    MIFID_II = "mifid_ii"
    GDPR = "gdpr"
    SOX = "sox"
    AML = "aml"
    KYC = "kyc"
    BEST_EXECUTION = "best_execution"
    TRADE_REPORTING = "trade_reporting"
    RECORD_KEEPING = "record_keeping"

class ComplianceLevel(Enum):
    """Compliance levels"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"

class ReportType(Enum):
    """Report types"""
    TRADE_REPORT = "trade_report"
    ORDER_REPORT = "order_report"
    POSITION_REPORT = "position_report"
    PNL_REPORT = "pnl_report"
    RISK_REPORT = "risk_report"
    COMPLIANCE_REPORT = "compliance_report"
    AUDIT_TRAIL = "audit_trail"

@dataclass
class ComplianceEvent:
    """Compliance event structure"""
    event_id: str
    regulation: RegulationType
    event_type: str
    level: ComplianceLevel
    symbol: str
    description: str
    timestamp: datetime
    data: Dict[str, Any]
    action_required: bool
    action_taken: bool = False
    resolved: bool = False

@dataclass
class TradeReport:
    """Trade report structure for regulatory reporting"""
    trade_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    execution_timestamp: datetime
    venue: str
    broker: str
    account_id: str
    order_id: str
    client_order_id: str
    capacity: str  # Principal, Agency, etc.
    settlement_date: datetime
    currency: str
    commission: float
    fees: Dict[str, float]
    regulatory_fields: Dict[str, Any]

@dataclass
class BestExecutionAnalysis:
    """Best execution analysis structure"""
    order_id: str
    symbol: str
    side: str
    quantity: int
    execution_price: float
    benchmark_prices: Dict[str, float]
    execution_venue: str
    alternative_venues: List[str]
    execution_quality_score: float
    market_impact: float
    timing_risk: float
    opportunity_cost: float
    best_execution_met: bool
    analysis_timestamp: datetime

class RegulatoryCompliance:
    """Production regulatory compliance system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.compliance_events = {}
        self.trade_reports = {}
        self.best_execution_analyses = {}
        self.running = False
        self.encryption_key = self._generate_encryption_key()
        self.db_connection = None
        self.audit_trail = []
        self.compliance_rules = {}
        self.reporting_schedule = {}
        
        # Initialize database
        self._initialize_database()
        
        # Initialize compliance rules
        self._initialize_compliance_rules()
        
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for sensitive data"""
        password = self.config.get('encryption_password', 'default_password').encode()
        salt = b'salt_for_compliance'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def _initialize_database(self):
        """Initialize compliance database"""
        try:
            self.db_connection = sqlite3.connect('compliance.db')
            cursor = self.db_connection.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS compliance_events (
                    event_id TEXT PRIMARY KEY,
                    regulation TEXT,
                    event_type TEXT,
                    level TEXT,
                    symbol TEXT,
                    description TEXT,
                    timestamp TEXT,
                    data TEXT,
                    action_required BOOLEAN,
                    action_taken BOOLEAN,
                    resolved BOOLEAN
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_reports (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    quantity INTEGER,
                    price REAL,
                    execution_timestamp TEXT,
                    venue TEXT,
                    broker TEXT,
                    account_id TEXT,
                    order_id TEXT,
                    client_order_id TEXT,
                    capacity TEXT,
                    settlement_date TEXT,
                    currency TEXT,
                    commission REAL,
                    fees TEXT,
                    regulatory_fields TEXT,
                    created_at TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS best_execution_analyses (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    quantity INTEGER,
                    execution_price REAL,
                    benchmark_prices TEXT,
                    execution_venue TEXT,
                    alternative_venues TEXT,
                    execution_quality_score REAL,
                    market_impact REAL,
                    timing_risk REAL,
                    opportunity_cost REAL,
                    best_execution_met BOOLEAN,
                    analysis_timestamp TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_trail (
                    entry_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    user_id TEXT,
                    action TEXT,
                    resource TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT
                )
            ''')
            
            self.db_connection.commit()
            logger.info("Compliance database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize compliance database: {e}")
            raise
    
    def _initialize_compliance_rules(self):
        """Initialize compliance rules"""
        self.compliance_rules = {
            RegulationType.FINRA: {
                'max_position_size': 0.20,  # 20% of portfolio
                'max_leverage': 2.0,
                'trade_reporting_required': True,
                'best_execution_required': True,
                'record_retention_years': 6
            },
            RegulationType.SEC: {
                'pattern_day_trader_threshold': 4,
                'pattern_day_trader_equity': 25000,
                'margin_requirement': 0.50,
                'trade_reporting_required': True,
                'record_retention_years': 7
            },
            RegulationType.MIFID_II: {
                'best_execution_required': True,
                'trade_reporting_required': True,
                'record_retention_years': 5,
                'client_classification_required': True,
                'timestamp_precision': 'microsecond'
            },
            RegulationType.GDPR: {
                'data_retention_years': 5,
                'consent_required': True,
                'data_subject_rights': True,
                'encryption_required': True
            },
            RegulationType.AML: {
                'suspicious_activity_threshold': 10000,
                'reporting_threshold': 10000,
                'customer_due_diligence_required': True,
                'transaction_monitoring_required': True
            }
        }
        
        # Initialize reporting schedule
        self.reporting_schedule = {
            ReportType.TRADE_REPORT: timedelta(hours=1),  # Hourly
            ReportType.POSITION_REPORT: timedelta(days=1),  # Daily
            ReportType.PNL_REPORT: timedelta(days=1),  # Daily
            ReportType.RISK_REPORT: timedelta(days=1),  # Daily
            ReportType.COMPLIANCE_REPORT: timedelta(days=7),  # Weekly
            ReportType.AUDIT_TRAIL: timedelta(hours=1)  # Hourly
        }
    
    async def start(self):
        """Start compliance system"""
        self.running = True
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_trade_reporting())
        asyncio.create_task(self._monitor_best_execution())
        asyncio.create_task(self._monitor_position_limits())
        asyncio.create_task(self._monitor_aml_requirements())
        asyncio.create_task(self._generate_reports())
        asyncio.create_task(self._maintain_audit_trail())
        
        logger.info("Regulatory compliance system started")
    
    async def stop(self):
        """Stop compliance system"""
        self.running = False
        
        if self.db_connection:
            self.db_connection.close()
        
        logger.info("Regulatory compliance system stopped")
    
    async def record_trade(self, trade_report: TradeReport) -> bool:
        """Record trade for regulatory reporting"""
        try:
            # Validate trade data
            if not self._validate_trade_data(trade_report):
                await self._create_compliance_event(
                    RegulationType.TRADE_REPORTING,
                    "INVALID_TRADE_DATA",
                    ComplianceLevel.WARNING,
                    trade_report.symbol,
                    f"Invalid trade data for {trade_report.trade_id}",
                    trade_report.__dict__
                )
                return False
            
            # Encrypt sensitive data
            encrypted_data = self._encrypt_sensitive_data(trade_report)
            
            # Store in database
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO trade_reports (
                    trade_id, symbol, side, quantity, price, execution_timestamp,
                    venue, broker, account_id, order_id, client_order_id,
                    capacity, settlement_date, currency, commission, fees,
                    regulatory_fields, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_report.trade_id,
                trade_report.symbol,
                trade_report.side,
                trade_report.quantity,
                trade_report.price,
                trade_report.execution_timestamp.isoformat(),
                trade_report.venue,
                trade_report.broker,
                trade_report.account_id,
                trade_report.order_id,
                trade_report.client_order_id,
                trade_report.capacity,
                trade_report.settlement_date.isoformat(),
                trade_report.currency,
                trade_report.commission,
                json.dumps(trade_report.fees),
                json.dumps(trade_report.regulatory_fields),
                datetime.utcnow().isoformat()
            ))
            
            self.db_connection.commit()
            
            # Store in memory
            self.trade_reports[trade_report.trade_id] = trade_report
            
            # Add to audit trail
            await self._add_audit_trail(
                "TRADE_RECORDED",
                f"Trade {trade_report.trade_id} recorded",
                {"trade_id": trade_report.trade_id, "symbol": trade_report.symbol}
            )
            
            logger.info(f"Trade recorded: {trade_report.trade_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
            await self._create_compliance_event(
                RegulationType.TRADE_REPORTING,
                "TRADE_RECORDING_ERROR",
                ComplianceLevel.CRITICAL,
                trade_report.symbol,
                f"Failed to record trade: {e}",
                trade_report.__dict__
            )
            return False
    
    async def analyze_best_execution(self, analysis: BestExecutionAnalysis) -> bool:
        """Analyze and store best execution"""
        try:
            # Validate analysis data
            if not self._validate_best_execution_data(analysis):
                await self._create_compliance_event(
                    RegulationType.BEST_EXECUTION,
                    "INVALID_BEST_EXECUTION_DATA",
                    ComplianceLevel.WARNING,
                    analysis.symbol,
                    f"Invalid best execution data for {analysis.order_id}",
                    analysis.__dict__
                )
                return False
            
            # Store in database
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO best_execution_analyses (
                    order_id, symbol, side, quantity, execution_price,
                    benchmark_prices, execution_venue, alternative_venues,
                    execution_quality_score, market_impact, timing_risk,
                    opportunity_cost, best_execution_met, analysis_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis.order_id,
                analysis.symbol,
                analysis.side,
                analysis.quantity,
                analysis.execution_price,
                json.dumps(analysis.benchmark_prices),
                analysis.execution_venue,
                json.dumps(analysis.alternative_venues),
                analysis.execution_quality_score,
                analysis.market_impact,
                analysis.timing_risk,
                analysis.opportunity_cost,
                analysis.best_execution_met,
                analysis.analysis_timestamp.isoformat()
            ))
            
            self.db_connection.commit()
            
            # Store in memory
            self.best_execution_analyses[analysis.order_id] = analysis
            
            # Check if best execution was met
            if not analysis.best_execution_met:
                await self._create_compliance_event(
                    RegulationType.BEST_EXECUTION,
                    "BEST_EXECUTION_NOT_MET",
                    ComplianceLevel.VIOLATION,
                    analysis.symbol,
                    f"Best execution not met for {analysis.order_id}",
                    analysis.__dict__
                )
            
            # Add to audit trail
            await self._add_audit_trail(
                "BEST_EXECUTION_ANALYZED",
                f"Best execution analysis for {analysis.order_id}",
                {"order_id": analysis.order_id, "symbol": analysis.symbol, "met": analysis.best_execution_met}
            )
            
            logger.info(f"Best execution analysis completed: {analysis.order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to analyze best execution: {e}")
            await self._create_compliance_event(
                RegulationType.BEST_EXECUTION,
                "BEST_EXECUTION_ERROR",
                ComplianceLevel.CRITICAL,
                analysis.symbol,
                f"Failed to analyze best execution: {e}",
                analysis.__dict__
            )
            return False
    
    def _validate_trade_data(self, trade_report: TradeReport) -> bool:
        """Validate trade data for compliance"""
        # Check required fields
        required_fields = [
            'trade_id', 'symbol', 'side', 'quantity', 'price',
            'execution_timestamp', 'venue', 'broker', 'account_id'
        ]
        
        for field in required_fields:
            if not hasattr(trade_report, field) or getattr(trade_report, field) is None:
                return False
        
        # Validate data types and ranges
        if trade_report.quantity <= 0:
            return False
        
        if trade_report.price <= 0:
            return False
        
        if trade_report.side not in ['buy', 'sell']:
            return False
        
        # Validate timestamp
        if isinstance(trade_report.execution_timestamp, str):
            try:
                datetime.fromisoformat(trade_report.execution_timestamp)
            except ValueError:
                return False
        
        return True
    
    def _validate_best_execution_data(self, analysis: BestExecutionAnalysis) -> bool:
        """Validate best execution data"""
        # Check required fields
        required_fields = [
            'order_id', 'symbol', 'side', 'quantity', 'execution_price',
            'execution_venue', 'execution_quality_score'
        ]
        
        for field in required_fields:
            if not hasattr(analysis, field) or getattr(analysis, field) is None:
                return False
        
        # Validate data types and ranges
        if analysis.quantity <= 0:
            return False
        
        if analysis.execution_price <= 0:
            return False
        
        if not 0 <= analysis.execution_quality_score <= 1:
            return False
        
        return True
    
    def _encrypt_sensitive_data(self, trade_report: TradeReport) -> Dict[str, Any]:
        """Encrypt sensitive trade data"""
        fernet = Fernet(self.encryption_key)
        
        # Fields to encrypt
        sensitive_fields = ['account_id', 'client_order_id']
        
        encrypted_data = trade_report.__dict__.copy()
        
        for field in sensitive_fields:
            if hasattr(trade_report, field):
                value = getattr(trade_report, field)
                if isinstance(value, str):
                    encrypted_value = fernet.encrypt(value.encode())
                    encrypted_data[field] = encrypted_value.decode()
        
        return encrypted_data
    
    async def _monitor_trade_reporting(self):
        """Monitor trade reporting compliance"""
        while self.running:
            try:
                # Check for unreported trades
                current_time = datetime.utcnow()
                reporting_deadline = current_time - timedelta(hours=1)
                
                cursor = self.db_connection.cursor()
                cursor.execute('''
                    SELECT trade_id, symbol, created_at FROM trade_reports
                    WHERE created_at < ?
                ''', (reporting_deadline.isoformat(),))
                
                overdue_trades = cursor.fetchall()
                
                for trade in overdue_trades:
                    await self._create_compliance_event(
                        RegulationType.TRADE_REPORTING,
                        "OVERDUE_TRADE_REPORT",
                        ComplianceLevel.WARNING,
                        trade[1],
                        f"Trade {trade[0]} reporting overdue",
                        {"trade_id": trade[0]}
                    )
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring trade reporting: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_best_execution(self):
        """Monitor best execution compliance"""
        while self.running:
            try:
                # Check for orders without best execution analysis
                cursor = self.db_connection.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM best_execution_analyses
                    WHERE best_execution_met = 0
                    AND analysis_timestamp > ?
                ''', ((datetime.utcnow() - timedelta(days=1)).isoformat(),))
                
                violations = cursor.fetchone()[0]
                
                if violations > 0:
                    await self._create_compliance_event(
                        RegulationType.BEST_EXECUTION,
                        "BEST_EXECUTION_VIOLATIONS",
                        ComplianceLevel.WARNING,
                        "MULTIPLE",
                        f"{violations} best execution violations in last 24 hours",
                        {"violations": violations}
                    )
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error monitoring best execution: {e}")
                await asyncio.sleep(300)
    
    async def _monitor_position_limits(self):
        """Monitor position limits compliance"""
        while self.running:
            try:
                # This would integrate with position data
                # Simplified implementation
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring position limits: {e}")
                await asyncio.sleep(120)
    
    async def _monitor_aml_requirements(self):
        """Monitor AML requirements"""
        while self.running:
            try:
                # Check for suspicious transactions
                cursor = self.db_connection.cursor()
                cursor.execute('''
                    SELECT trade_id, symbol, price, quantity FROM trade_reports
                    WHERE price * quantity > ?
                    AND created_at > ?
                ''', (
                    self.compliance_rules[RegulationType.AML]['suspicious_activity_threshold'],
                    (datetime.utcnow() - timedelta(days=1)).isoformat()
                ))
                
                suspicious_trades = cursor.fetchall()
                
                for trade in suspicious_trades:
                    await self._create_compliance_event(
                        RegulationType.AML,
                        "SUSPICIOUS_TRANSACTION",
                        ComplianceLevel.WARNING,
                        trade[1],
                        f"Suspicious transaction: {trade[0]}",
                        {"trade_id": trade[0], "value": trade[2] * trade[3]}
                    )
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring AML requirements: {e}")
                await asyncio.sleep(300)
    
    async def _generate_reports(self):
        """Generate regulatory reports"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                # Generate different types of reports
                for report_type, frequency in self.reporting_schedule.items():
                    if self._should_generate_report(report_type, frequency, current_time):
                        await self._generate_report(report_type, current_time)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error generating reports: {e}")
                await asyncio.sleep(60)
    
    def _should_generate_report(self, report_type: ReportType, frequency: timedelta, current_time: datetime) -> bool:
        """Check if report should be generated"""
        # Simplified implementation
        # In production, would track last generation time
        return True
    
    async def _generate_report(self, report_type: ReportType, current_time: datetime):
        """Generate specific regulatory report"""
        try:
            if report_type == ReportType.TRADE_REPORT:
                await self._generate_trade_report(current_time)
            elif report_type == ReportType.POSITION_REPORT:
                await self._generate_position_report(current_time)
            elif report_type == ReportType.PNL_REPORT:
                await self._generate_pnl_report(current_time)
            elif report_type == ReportType.COMPLIANCE_REPORT:
                await self._generate_compliance_report(current_time)
            
        except Exception as e:
            logger.error(f"Error generating {report_type.value} report: {e}")
    
    async def _generate_trade_report(self, current_time: datetime):
        """Generate trade report"""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            SELECT * FROM trade_reports
            WHERE created_at > ?
            ORDER BY execution_timestamp
        ''', ((current_time - timedelta(hours=1)).isoformat(),))
        
        trades = cursor.fetchall()
        
        # Generate report in required format
        report_data = {
            'report_type': 'trade_report',
            'period_start': (current_time - timedelta(hours=1)).isoformat(),
            'period_end': current_time.isoformat(),
            'trades': [
                {
                    'trade_id': trade[0],
                    'symbol': trade[1],
                    'side': trade[2],
                    'quantity': trade[3],
                    'price': trade[4],
                    'execution_timestamp': trade[5],
                    'venue': trade[6],
                    'broker': trade[7]
                }
                for trade in trades
            ]
        }
        
        # Store report
        await self._store_report(report_type, report_data)
        
        logger.info(f"Trade report generated with {len(trades)} trades")
    
    async def _generate_position_report(self, current_time: datetime):
        """Generate position report"""
        # Implementation for position reporting
        pass
    
    async def _generate_pnl_report(self, current_time: datetime):
        """Generate P&L report"""
        # Implementation for P&L reporting
        pass
    
    async def _generate_compliance_report(self, current_time: datetime):
        """Generate compliance report"""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            SELECT regulation, level, COUNT(*) as count
            FROM compliance_events
            WHERE timestamp > ?
            GROUP BY regulation, level
        ''', ((current_time - timedelta(days=7)).isoformat(),))
        
        events = cursor.fetchall()
        
        report_data = {
            'report_type': 'compliance_report',
            'period_start': (current_time - timedelta(days=7)).isoformat(),
            'period_end': current_time.isoformat(),
            'events': [
                {
                    'regulation': event[0],
                    'level': event[1],
                    'count': event[2]
                }
                for event in events
            ]
        }
        
        await self._store_report(ReportType.COMPLIANCE_REPORT, report_data)
        
        logger.info("Compliance report generated")
    
    async def _store_report(self, report_type: ReportType, report_data: Dict[str, Any]):
        """Store report data"""
        # Implementation for storing reports
        # Could be file system, database, or external API
        pass
    
    async def _maintain_audit_trail(self):
        """Maintain audit trail"""
        while self.running:
            try:
                # Clean old audit trail entries
                retention_days = self.compliance_rules.get(RegulationType.GDPR, {}).get('data_retention_years', 5)
                cutoff_date = datetime.utcnow() - timedelta(days=retention_days * 365)
                
                cursor = self.db_connection.cursor()
                cursor.execute('''
                    DELETE FROM audit_trail
                    WHERE timestamp < ?
                ''', (cutoff_date.isoformat(),))
                
                self.db_connection.commit()
                
                await asyncio.sleep(86400)  # Daily maintenance
                
            except Exception as e:
                logger.error(f"Error maintaining audit trail: {e}")
                await asyncio.sleep(3600)
    
    async def _add_audit_trail(self, action: str, resource: str, details: Dict[str, Any]):
        """Add entry to audit trail"""
        try:
            entry_id = str(uuid.uuid4())
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO audit_trail (
                    entry_id, timestamp, user_id, action, resource, details,
                    ip_address, user_agent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry_id,
                datetime.utcnow().isoformat(),
                "system",  # Would be actual user ID
                action,
                resource,
                json.dumps(details),
                "127.0.0.1",  # Would be actual IP
                "MiniQuantFund/1.0"  # Would be actual user agent
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to add audit trail entry: {e}")
    
    async def _create_compliance_event(self, regulation: RegulationType, event_type: str, 
                                     level: ComplianceLevel, symbol: str, description: str, 
                                     data: Dict[str, Any]):
        """Create compliance event"""
        event = ComplianceEvent(
            event_id=str(uuid.uuid4()),
            regulation=regulation,
            event_type=event_type,
            level=level,
            symbol=symbol,
            description=description,
            timestamp=datetime.utcnow(),
            data=data,
            action_required=level in [ComplianceLevel.WARNING, ComplianceLevel.VIOLATION, ComplianceLevel.CRITICAL]
        )
        
        self.compliance_events[event.event_id] = event
        
        # Store in database
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO compliance_events (
                event_id, regulation, event_type, level, symbol, description,
                timestamp, data, action_required, action_taken, resolved
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id,
            event.regulation.value,
            event.event_type,
            event.level.value,
            event.symbol,
            event.description,
            event.timestamp.isoformat(),
            json.dumps(event.data),
            event.action_required,
            event.action_taken,
            event.resolved
        ))
        
        self.db_connection.commit()
        
        logger.warning(f"Compliance event created: {event.event_id} - {description}")
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance summary"""
        cursor = self.db_connection.cursor()
        
        # Get event counts by level
        cursor.execute('''
            SELECT level, COUNT(*) as count
            FROM compliance_events
            WHERE timestamp > ?
            GROUP BY level
        ''', ((datetime.utcnow() - timedelta(days=1)).isoformat(),))
        
        events_by_level = dict(cursor.fetchall())
        
        # Get trade report count
        cursor.execute('''
            SELECT COUNT(*) FROM trade_reports
            WHERE created_at > ?
        ''', ((datetime.utcnow() - timedelta(days=1)).isoformat(),))
        
        trade_reports_count = cursor.fetchone()[0]
        
        # Get best execution compliance rate
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN best_execution_met = 1 THEN 1 ELSE 0 END) as compliant
            FROM best_execution_analyses
            WHERE analysis_timestamp > ?
        ''', ((datetime.utcnow() - timedelta(days=1)).isoformat(),))
        
        total_analyses, compliant_analyses = cursor.fetchone()
        best_execution_rate = compliant_analyses / total_analyses if total_analyses > 0 else 0
        
        return {
            'events_by_level': events_by_level,
            'trade_reports_count': trade_reports_count,
            'best_execution_rate': best_execution_rate,
            'total_compliance_events': len(self.compliance_events),
            'active_alerts': len([e for e in self.compliance_events.values() if not e.resolved])
        }

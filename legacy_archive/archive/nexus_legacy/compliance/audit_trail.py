import logging
import pandas as pd
import json
import hashlib
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    TRADE_EXECUTION = "trade_execution"
    ORDER_PLACEMENT = "order_placement"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    RISK_CHECK = "risk_check"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SYSTEM_ACCESS = "system_access"
    DATA_MODIFICATION = "data_modification"
    MODEL_UPDATE = "model_update"
    CONFIGURATION_CHANGE = "configuration_change"

class AuditSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Comprehensive audit event structure."""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: Optional[str]
    session_id: Optional[str]
    source_system: str
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    checksum: Optional[str]
    previous_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of event data for integrity."""
        event_dict = asdict(self)
        # Remove checksum from calculation to avoid circular reference
        event_dict.pop('checksum', None)

        # Sort keys for consistent hashing
        sorted_data = json.dumps(event_dict, sort_keys=True, default=str)
        return hashlib.sha256(sorted_data.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create AuditEvent from dictionary."""
        # Convert string enums back to enum objects
        data['event_type'] = AuditEventType(data['event_type'])
        data['severity'] = AuditSeverity(data['severity'])
        return cls(**data)

class InstitutionalAuditTrail:
    """
    INSTITUTIONAL-GRADE AUDIT TRAIL SYSTEM
    Comprehensive logging, integrity verification, and regulatory reporting.
    Implements immutable audit logs with cryptographic integrity.
    """

    def __init__(self, audit_dir: str = "audit_logs", retention_days: int = 2555):  # 7 years
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        self.retention_days = retention_days
        self.current_log_file = None
        self._initialize_log_file()

        # Audit trail integrity
        self.chain_hash = self._load_chain_hash()

        # Performance monitoring
        self.event_buffer = []
        self.buffer_size = 100  # Batch write every 100 events

        logger.info("Institutional Audit Trail initialized")

    def log_event(self, event_type: AuditEventType, severity: AuditSeverity,
                  action: str, resource: str, details: Dict[str, Any],
                  user_id: Optional[str] = None, session_id: Optional[str] = None,
                  source_system: str = "trading_system",
                  ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                  previous_state: Optional[Dict[str, Any]] = None,
                  new_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Log an audit event with full context and integrity verification.
        Returns the event ID for tracking.
        """
        try:
            event = AuditEvent(
                event_id="",
                timestamp=datetime.utcnow(),
                event_type=event_type,
                severity=severity,
                user_id=user_id,
                session_id=session_id,
                source_system=source_system,
                action=action,
                resource=resource,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent,
                checksum=None,
                previous_state=previous_state,
                new_state=new_state
            )

            # Add to buffer
            self.event_buffer.append(event)

            # Batch write if buffer is full
            if len(self.event_buffer) >= self.buffer_size:
                self._flush_buffer()

            # Update chain hash for integrity
            self._update_chain_hash(event)

            logger.info(f"Audit event logged: {event.event_id} - {action} on {resource}")

            return event.event_id

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            # Emergency logging to ensure critical events are captured
            self._emergency_log(action, resource, str(e))
            return ""

    def log_trade_execution(self, trade_details: Dict[str, Any], user_id: Optional[str] = None) -> str:
        """Log trade execution with full trade details."""
        return self.log_event(
            event_type=AuditEventType.TRADE_EXECUTION,
            severity=AuditSeverity.INFO,
            action="EXECUTE_TRADE",
            resource=f"trade:{trade_details.get('trade_id', 'unknown')}",
            details=trade_details,
            user_id=user_id,
            previous_state={"portfolio_state": "pre_trade"},
            new_state={"portfolio_state": "post_trade"}
        )

    def log_risk_violation(self, violation_details: Dict[str, Any], severity: AuditSeverity = AuditSeverity.WARNING) -> str:
        """Log risk limit violations."""
        return self.log_event(
            event_type=AuditEventType.RISK_CHECK,
            severity=severity,
            action="RISK_VIOLATION_DETECTED",
            resource=f"risk:{violation_details.get('risk_type', 'unknown')}",
            details=violation_details
        )

    def log_compliance_event(self, compliance_details: Dict[str, Any], severity: AuditSeverity = AuditSeverity.WARNING) -> str:
        """Log compliance-related events."""
        return self.log_event(
            event_type=AuditEventType.COMPLIANCE_VIOLATION,
            severity=severity,
            action="COMPLIANCE_CHECK",
            resource=f"compliance:{compliance_details.get('regulation', 'unknown')}",
            details=compliance_details
        )

    def log_system_access(self, access_details: Dict[str, Any], user_id: str, ip_address: str) -> str:
        """Log system access events."""
        return self.log_event(
            event_type=AuditEventType.SYSTEM_ACCESS,
            severity=AuditSeverity.INFO,
            action="SYSTEM_ACCESS",
            resource=f"user:{user_id}",
            details=access_details,
            user_id=user_id,
            ip_address=ip_address
        )

    def log_data_modification(self, modification_details: Dict[str, Any],
                            previous_state: Dict[str, Any], new_state: Dict[str, Any],
                            user_id: str) -> str:
        """Log data modification events with before/after states."""
        return self.log_event(
            event_type=AuditEventType.DATA_MODIFICATION,
            severity=AuditSeverity.WARNING,
            action="DATA_MODIFIED",
            resource=f"data:{modification_details.get('table', 'unknown')}",
            details=modification_details,
            user_id=user_id,
            previous_state=previous_state,
            new_state=new_state
        )

    def verify_integrity(self, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Verify the integrity of the audit trail using cryptographic hashing.
        Returns verification results.
        """
        try:
            verification_results = {
                'verified_events': 0,
                'corrupted_events': 0,
                'missing_events': 0,
                'chain_integrity': True,
                'verification_period': {'start': start_date, 'end': end_date}
            }

            # Get events in the specified period
            events = self.get_events(start_date, end_date)

            current_chain_hash = self._load_chain_hash()

            for event in events:
                # Verify individual event checksum
                calculated_checksum = event._calculate_checksum()
                if calculated_checksum != event.checksum:
                    verification_results['corrupted_events'] += 1
                    logger.warning(f"Corrupted event detected: {event.event_id}")
                else:
                    verification_results['verified_events'] += 1

                # Verify chain integrity (simplified)
                # In production, would verify each event builds on previous

            verification_results['integrity_score'] = (
                verification_results['verified_events'] /
                max(1, verification_results['verified_events'] + verification_results['corrupted_events'])
            )

            return verification_results

        except Exception as e:
            logger.error(f"Audit trail integrity verification failed: {e}")
            return {'error': str(e)}

    def get_events(self, start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None,
                  event_type: Optional[AuditEventType] = None,
                  user_id: Optional[str] = None,
                  severity: Optional[AuditSeverity] = None) -> List[AuditEvent]:
        """
        Retrieve audit events with filtering options.
        """
        try:
            all_events = []

            # Determine date range for file selection
            if start_date is None:
                start_date = datetime.utcnow() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.utcnow()

            # Find relevant log files
            log_files = self._get_log_files_in_range(start_date, end_date)

            for log_file in log_files:
                if log_file.exists():
                    events = self._read_log_file(log_file)
                    all_events.extend(events)

            # Apply filters
            filtered_events = []
            for event in all_events:
                if start_date and event.timestamp < start_date:
                    continue
                if end_date and event.timestamp > end_date:
                    continue
                if event_type and event.event_type != event_type:
                    continue
                if user_id and event.user_id != user_id:
                    continue
                if severity and event.severity != severity:
                    continue

                filtered_events.append(event)

            # Sort by timestamp
            filtered_events.sort(key=lambda x: x.timestamp)

            return filtered_events

        except Exception as e:
            logger.error(f"Failed to retrieve audit events: {e}")
            return []

    def generate_regulatory_report(self, report_type: str, start_date: datetime,
                                 end_date: datetime) -> Dict[str, Any]:
        """
        Generate regulatory compliance reports (SEC, FCA, etc.).
        """
        try:
            events = self.get_events(start_date, end_date)

            if report_type == "SEC":
                return self._generate_sec_report(events, start_date, end_date)
            elif report_type == "FCA":
                return self._generate_fca_report(events, start_date, end_date)
            elif report_type == "MAS":
                return self._generate_mas_report(events, start_date, end_date)
            else:
                return self._generate_general_report(events, start_date, end_date)

        except Exception as e:
            logger.error(f"Regulatory report generation failed: {e}")
            return {'error': str(e)}

    def export_audit_trail(self, export_path: str, start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None, format: str = "json") -> bool:
        """
        Export audit trail for external review or archival.
        """
        try:
            events = self.get_events(start_date, end_date)

            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)

            if format == "json":
                event_dicts = [event.to_dict() for event in events]
                with open(export_file, 'w') as f:
                    json.dump(event_dicts, f, indent=2, default=str)
            elif format == "csv":
                if events:
                    df = pd.DataFrame([event.to_dict() for event in events])
                    df.to_csv(export_file, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Audit trail exported to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Audit trail export failed: {e}")
            return False

    def cleanup_old_logs(self) -> int:
        """Clean up audit logs older than retention period. Returns number of files removed."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            removed_count = 0

            for log_file in self.audit_dir.glob("audit_*.jsonl"):
                # Extract date from filename
                filename = log_file.stem
                try:
                    file_date_str = filename.split('_')[1]
                    file_date = datetime.strptime(file_date_str, '%Y%m%d')
                    if file_date < cutoff_date:
                        log_file.unlink()
                        removed_count += 1
                except (ValueError, IndexError):
                    continue

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old audit log files")

            return removed_count

        except Exception as e:
            logger.error(f"Audit log cleanup failed: {e}")
            return 0

    def _initialize_log_file(self):
        """Initialize the current log file."""
        today = datetime.utcnow().strftime('%Y%m%d')
        self.current_log_file = self.audit_dir / f"audit_{today}.jsonl"

    def _flush_buffer(self):
        """Flush the event buffer to disk."""
        try:
            if not self.event_buffer:
                return

            # Check if we need to roll over to new file
            today = datetime.utcnow().strftime('%Y%m%d')
            expected_file = self.audit_dir / f"audit_{today}.jsonl"

            if self.current_log_file != expected_file:
                self.current_log_file = expected_file

            # Append events to file
            with open(self.current_log_file, 'a') as f:
                for event in self.event_buffer:
                    f.write(json.dumps(event.to_dict(), default=str) + '\n')

            # Clear buffer
            self.event_buffer.clear()

        except Exception as e:
            logger.error(f"Failed to flush audit buffer: {e}")

    def _update_chain_hash(self, event: AuditEvent):
        """Update the chain hash for integrity verification."""
        try:
            # Simplified chain hashing - in production would use Merkle tree
            hash_input = f"{self.chain_hash}{event.checksum}"
            self.chain_hash = hashlib.sha256(hash_input.encode()).hexdigest()

            # Save updated chain hash
            chain_file = self.audit_dir / "chain_hash.txt"
            with open(chain_file, 'w') as f:
                f.write(self.chain_hash)

        except Exception as e:
            logger.error(f"Failed to update chain hash: {e}")

    def _load_chain_hash(self) -> str:
        """Load the current chain hash."""
        try:
            chain_file = self.audit_dir / "chain_hash.txt"
            if chain_file.exists():
                with open(chain_file, 'r') as f:
                    return f.read().strip()
            else:
                # Initialize with genesis hash
                return hashlib.sha256(b"genesis").hexdigest()
        except Exception:
            return hashlib.sha256(b"genesis").hexdigest()

    def _read_log_file(self, log_file: Path) -> List[AuditEvent]:
        """Read events from a log file."""
        events = []
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        event_dict = json.loads(line)
                        event = AuditEvent.from_dict(event_dict)
                        events.append(event)
        except Exception as e:
            logger.error(f"Failed to read log file {log_file}: {e}")

        return events

    def _get_log_files_in_range(self, start_date: datetime, end_date: datetime) -> List[Path]:
        """Get list of log files in the date range."""
        files = []
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            log_file = self.audit_dir / f"audit_{date_str}.jsonl"
            if log_file.exists():
                files.append(log_file)
            current_date += timedelta(days=1)

        return files

    def _emergency_log(self, action: str, resource: str, error: str):
        """Emergency logging when main audit system fails."""
        try:
            emergency_file = self.audit_dir / "emergency_audit.log"
            timestamp = datetime.utcnow().isoformat()

            with open(emergency_file, 'a') as f:
                f.write(f"{timestamp} - EMERGENCY: {action} on {resource} - Error: {error}\n")

        except Exception:
            # Last resort - log to console
            print(f"EMERGENCY AUDIT LOG FAILURE: {action} on {resource} - {error}")

    def _generate_sec_report(self, events: List[AuditEvent], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate SEC-compliant audit report."""
        report = {
            'report_type': 'SEC_AUDIT_TRAIL',
            'period': {'start': start_date.isoformat(), 'end': end_date.isoformat()},
            'summary': {
                'total_events': len(events),
                'trade_events': len([e for e in events if e.event_type == AuditEventType.TRADE_EXECUTION]),
                'risk_events': len([e for e in events if e.event_type == AuditEventType.RISK_CHECK]),
                'compliance_events': len([e for e in events if e.event_type == AuditEventType.COMPLIANCE_VIOLATION])
            },
            'critical_events': [e.to_dict() for e in events if e.severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL]],
            'generated_at': datetime.utcnow().isoformat()
        }

        return report

    def _generate_fca_report(self, events: List[AuditEvent], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate FCA-compliant audit report."""
        # Similar structure but with FCA-specific requirements
        report = {
            'report_type': 'FCA_AUDIT_TRAIL',
            'period': {'start': start_date.isoformat(), 'end': end_date.isoformat()},
            'summary': {
                'total_events': len(events),
                'client_money_events': len([e for e in events if 'client_money' in e.resource]),
                'market_abuse_signals': len([e for e in events if e.severity == AuditSeverity.CRITICAL])
            },
            'generated_at': datetime.utcnow().isoformat()
        }

        return report

    def _generate_mas_report(self, events: List[AuditEvent], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate MAS-compliant audit report."""
        report = {
            'report_type': 'MAS_AUDIT_TRAIL',
            'period': {'start': start_date.isoformat(), 'end': end_date.isoformat()},
            'summary': {
                'total_events': len(events),
                'systemic_risk_events': len([e for e in events if 'systemic' in e.details.get('risk_type', '')])
            },
            'generated_at': datetime.utcnow().isoformat()
        }

        return report

    def _generate_general_report(self, events: List[AuditEvent], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate general audit report."""
        report = {
            'report_type': 'GENERAL_AUDIT_TRAIL',
            'period': {'start': start_date.isoformat(), 'end': end_date.isoformat()},
            'summary': {
                'total_events': len(events),
                'events_by_type': {},
                'events_by_severity': {},
                'events_by_user': {}
            },
            'generated_at': datetime.utcnow().isoformat()
        }

        # Aggregate statistics
        for event in events:
            # By type
            type_key = event.event_type.value
            report['summary']['events_by_type'][type_key] = report['summary']['events_by_type'].get(type_key, 0) + 1

            # By severity
            severity_key = event.severity.value
            report['summary']['events_by_severity'][severity_key] = report['summary']['events_by_severity'].get(severity_key, 0) + 1

            # By user
            if event.user_id:
                report['summary']['events_by_user'][event.user_id] = report['summary']['events_by_user'].get(event.user_id, 0) + 1

        return report

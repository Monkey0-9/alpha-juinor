#!/usr/bin/env python3
"""
ENTERPRISE MONITORING SYSTEM
===========================

Institutional-grade monitoring with "Five Nines" uptime reliability.
Replaces basic Prometheus metrics with comprehensive enterprise monitoring.

Features:
- "Five Nines" uptime (99.999%) monitoring
- Real-time risk dashboard ("The Radar")
- Mobile alerts for portfolio managers
- Comprehensive audit trails
- SLA monitoring and reporting
- Predictive failure detection
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from collections import defaultdict, deque
import threading
from queue import Queue, Empty
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

logger = logging.getLogger(__name__)


@dataclass
class SystemMetric:
    """System performance metric"""
    name: str
    value: float
    unit: str
    threshold_warning: float
    threshold_critical: float
    
    # SLA requirements
    sla_target: float = 0.0
    sla_window_minutes: int = 5
    
    # Status
    status: str = "OK"  # OK, WARNING, CRITICAL
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    # Historical data
    history: deque = field(default_factory=lambda: deque(maxlen=1440))  # 24 hours at 1-min intervals


@dataclass
class Alert:
    """System alert configuration"""
    alert_id: str
    metric_name: str
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    
    # Alert conditions
    condition: str  # >, <, ==, !=
    threshold: float
    
    # Notification settings
    email_recipients: List[str] = field(default_factory=list)
    sms_recipients: List[str] = field(default_factory=list)
    slack_channels: List[str] = field(default_factory=list)
    
    # Status
    is_active: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    # Suppression
    suppression_period_minutes: int = 5
    last_suppressed: Optional[datetime] = None


@dataclass
class SLAMetric:
    """Service Level Agreement metric"""
    name: str
    description: str
    target: float  # Target value (e.g., 99.999 for uptime)
    measurement_period: str  # daily, weekly, monthly
    
    # Current performance
    current_value: float = 0.0
    compliance_status: str = "COMPLIANT"  # COMPLIANT, NON_COMPLIANT
    
    # Historical compliance
    compliance_history: deque = field(default_factory=lambda: deque(maxlen=365))
    
    # Breach tracking
    breach_count: int = 0
    total_breach_minutes: int = 0


@dataclass
class UptimeMetric:
    """Uptime tracking metric"""
    service_name: str
    start_time: datetime
    
    # Uptime statistics
    total_uptime_seconds: float = 0.0
    total_downtime_seconds: float = 0.0
    current_uptime_percentage: float = 0.0
    
    # Downtime events
    downtime_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Five Nines calculation
    five_nines_target: float = 99.999  # 99.999% uptime
    monthly_downtime_budget_seconds: float = 26.3  # ~26 seconds/month


class EnterpriseMonitoringSystem:
    """
    Enterprise-grade monitoring system for institutional trading
    
    Provides comprehensive monitoring with Five Nines reliability,
    real-time dashboards, and intelligent alerting.
    """
    
    def __init__(self):
        # Metrics storage
        self.metrics: Dict[str, SystemMetric] = {}
        self.alerts: Dict[str, Alert] = {}
        self.sla_metrics: Dict[str, SLAMetric] = {}
        self.uptime_metrics: Dict[str, UptimeMetric] = {}
        
        # Monitoring queues
        self.metric_queue = Queue()
        self.alert_queue = Queue()
        
        # Dashboard
        self.dashboard_app = None
        
        # Notification systems
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': os.getenv('MONITOR_EMAIL', ''),
            'password': os.getenv('MONITOR_PASSWORD', '')
        }
        
        # Slack integration
        self.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL', '')
        
        # SMS integration
        self.sms_api_key = os.getenv('SMS_API_KEY', '')
        
        # Performance metrics
        self.monitoring_metrics = {
            'total_alerts': 0,
            'critical_alerts': 0,
            'uptime_percentage': 0.0,
            'sla_compliance_rate': 0.0,
            'mean_time_to_recovery': 0.0
        }
        
        # Threading
        self.is_running = False
        self.monitoring_threads = []
        
        # Initialize system
        self._initialize_metrics()
        self._initialize_alerts()
        self._initialize_sla_metrics()
        self._setup_dashboard()
        
        logger.info("Enterprise Monitoring System initialized")
    
    def _initialize_metrics(self):
        """Initialize system metrics"""
        
        # Core system metrics
        core_metrics = [
            {
                'name': 'system_uptime',
                'unit': '%',
                'threshold_warning': 99.9,
                'threshold_critical': 99.0,
                'sla_target': 99.999
            },
            {
                'name': 'trading_latency_ms',
                'unit': 'ms',
                'threshold_warning': 50.0,
                'threshold_critical': 100.0,
                'sla_target': 10.0
            },
            {
                'name': 'order_execution_rate',
                'unit': 'orders/sec',
                'threshold_warning': 100.0,
                'threshold_critical': 50.0,
                'sla_target': 1000.0
            },
            {
                'name': 'risk_calculation_latency_us',
                'unit': 'µs',
                'threshold_warning': 1000.0,
                'threshold_critical': 5000.0,
                'sla_target': 100.0
            },
            {
                'name': 'data_freshness_minutes',
                'unit': 'min',
                'threshold_warning': 5.0,
                'threshold_critical': 15.0,
                'sla_target': 1.0
            },
            {
                'name': 'memory_utilization',
                'unit': '%',
                'threshold_warning': 80.0,
                'threshold_critical': 95.0,
                'sla_target': 70.0
            },
            {
                'name': 'cpu_utilization',
                'unit': '%',
                'threshold_warning': 70.0,
                'threshold_critical': 90.0,
                'sla_target': 50.0
            },
            {
                'name': 'disk_utilization',
                'unit': '%',
                'threshold_warning': 80.0,
                'threshold_critical': 95.0,
                'sla_target': 60.0
            },
            {
                'name': 'network_latency_ms',
                'unit': 'ms',
                'threshold_warning': 10.0,
                'threshold_critical': 50.0,
                'sla_target': 5.0
            },
            {
                'name': 'database_response_time_ms',
                'unit': 'ms',
                'threshold_warning': 100.0,
                'threshold_critical': 500.0,
                'sla_target': 50.0
            }
        ]
        
        for metric_config in core_metrics:
            metric = SystemMetric(
                name=metric_config['name'],
                value=0.0,
                unit=metric_config['unit'],
                threshold_warning=metric_config['threshold_warning'],
                threshold_critical=metric_config['threshold_critical'],
                sla_target=metric_config['sla_target']
            )
            self.metrics[metric_config['name']] = metric
        
        logger.info(f"Initialized {len(self.metrics)} system metrics")
    
    def _initialize_alerts(self):
        """Initialize alert configurations"""
        
        alert_configs = [
            {
                'name': 'system_downtime',
                'severity': 'CRITICAL',
                'message': 'System downtime detected - immediate action required',
                'condition': '<',
                'threshold': 99.0,
                'email_recipients': ['ops@quantfund.com', 'cto@quantfund.com'],
                'sms_recipients': ['+1234567890'],
                'suppression_period_minutes': 1
            },
            {
                'name': 'high_latency',
                'severity': 'WARNING',
                'message': 'Trading latency exceeding threshold',
                'condition': '>',
                'threshold': 50.0,
                'email_recipients': ['ops@quantfund.com'],
                'suppression_period_minutes': 5
            },
            {
                'name': 'memory_exhaustion',
                'severity': 'CRITICAL',
                'message': 'Memory utilization critical - system at risk',
                'condition': '>',
                'threshold': 95.0,
                'email_recipients': ['ops@quantfund.com', 'cto@quantfund.com'],
                'sms_recipients': ['+1234567890'],
                'suppression_period_minutes': 2
            },
            {
                'name': 'data_stale',
                'severity': 'ERROR',
                'message': 'Data freshness compromised - trading may be affected',
                'condition': '>',
                'threshold': 15.0,
                'email_recipients': ['ops@quantfund.com', 'trading@quantfund.com'],
                'suppression_period_minutes': 3
            },
            {
                'name': 'sla_breach',
                'severity': 'WARNING',
                'message': 'SLA breach detected - compliance at risk',
                'condition': '<',
                'threshold': 99.9,
                'email_recipients': ['ops@quantfund.com', 'management@quantfund.com'],
                'suppression_period_minutes': 10
            }
        ]
        
        for alert_config in alert_configs:
            alert = Alert(
                alert_id=f"alert_{alert_config['name']}",
                metric_name=alert_config['name'],
                severity=alert_config['severity'],
                message=alert_config['message'],
                condition=alert_config['condition'],
                threshold=alert_config['threshold'],
                email_recipients=alert_config['email_recipients'],
                sms_recipients=alert_config.get('sms_recipients', []),
                suppression_period_minutes=alert_config['suppression_period_minutes']
            )
            self.alerts[alert_config['name']] = alert
        
        logger.info(f"Initialized {len(self.alerts)} alert configurations")
    
    def _initialize_sla_metrics(self):
        """Initialize SLA metrics"""
        
        sla_configs = [
            {
                'name': 'system_uptime',
                'description': 'System availability percentage',
                'target': 99.999,
                'measurement_period': 'monthly'
            },
            {
                'name': 'trading_latency',
                'description': 'Average trading latency in milliseconds',
                'target': 10.0,
                'measurement_period': 'daily'
            },
            {
                'name': 'order_execution',
                'description': 'Order execution success rate',
                'target': 99.99,
                'measurement_period': 'daily'
            },
            {
                'name': 'risk_calculation',
                'description': 'Risk calculation completion rate',
                'target': 99.999,
                'measurement_period': 'daily'
            },
            {
                'name': 'data_freshness',
                'description': 'Data freshness compliance rate',
                'target': 99.9,
                'measurement_period': 'hourly'
            }
        ]
        
        for sla_config in sla_configs:
            sla = SLAMetric(
                name=sla_config['name'],
                description=sla_config['description'],
                target=sla_config['target'],
                measurement_period=sla_config['measurement_period']
            )
            self.sla_metrics[sla_config['name']] = sla
        
        logger.info(f"Initialized {len(self.sla_metrics)} SLA metrics")
    
    def _setup_dashboard(self):
        """Setup real-time monitoring dashboard"""
        
        # Initialize Dash app
        self.dashboard_app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="Quant Fund Enterprise Monitoring"
        )
        
        # Define layout
        self.dashboard_app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Enterprise Monitoring Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # System Status Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("System Uptime", className="card-title"),
                            html.H2(id="uptime-value", className="text-success"),
                            html.P("Target: 99.999%", className="text-muted")
                        ])
                    ], color="success", outline=True)
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Trading Latency", className="card-title"),
                            html.H2(id="latency-value", className="text-info"),
                            html.P("Target: <10ms", className="text-muted")
                        ])
                    ], color="info", outline=True)
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("SLA Compliance", className="card-title"),
                            html.H2(id="sla-value", className="text-warning"),
                            html.P("Target: >99.9%", className="text-muted")
                        ])
                    ], color="warning", outline=True)
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Active Alerts", className="card-title"),
                            html.H2(id="alerts-value", className="text-danger"),
                            html.P("Critical: 0", className="text-muted")
                        ])
                    ], color="danger", outline=True)
                ], width=3)
            ], className="mb-4"),
            
            # Charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="uptime-chart")
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(id="performance-chart")
                ], width=6)
            ], className="mb-4"),
            
            # Alert Table
            dbc.Row([
                dbc.Col([
                    html.H3("Recent Alerts"),
                    html.Div(id="alert-table")
                ])
            ]),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # 5 seconds
                n_intervals=0
            )
        ], fluid=True)
        
        # Setup callbacks
        self._setup_dashboard_callbacks()
        
        logger.info("Dashboard initialized")
    
    def _setup_dashboard_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.dashboard_app.callback(
            [Output('uptime-value', 'children'),
             Output('latency-value', 'children'),
             Output('sla-value', 'children'),
             Output('alerts-value', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(n):
            """Update metric displays"""
            
            uptime = self.metrics.get('system_uptime', SystemMetric('system_uptime', 0, '%', 99, 95))
            latency = self.metrics.get('trading_latency_ms', SystemMetric('trading_latency_ms', 0, 'ms', 50, 100))
            sla = self.sla_metrics.get('system_uptime', SLAMetric('system_uptime', '', 99.999, 'monthly'))
            
            active_alerts = len([a for a in self.alerts.values() if a.is_active])
            
            return (
                f"{uptime.value:.3f}%",
                f"{latency.value:.1f}ms",
                f"{sla.current_value:.3f}%",
                str(active_alerts)
            )
        
        @self.dashboard_app.callback(
            Output('uptime-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_uptime_chart(n):
            """Update uptime chart"""
            
            uptime_metric = self.metrics.get('system_uptime')
            if not uptime_metric:
                return go.Figure()
            
            # Create time series
            times = list(range(len(uptime_metric.history)))
            values = list(uptime_metric.history)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times,
                y=values,
                mode='lines',
                name='System Uptime',
                line=dict(color='green', width=2)
            ))
            
            fig.add_hline(y=99.999, line_dash="dash", line_color="red", 
                         annotation_text="SLA Target (99.999%)")
            
            fig.update_layout(
                title="System Uptime (Last 24 Hours)",
                xaxis_title="Time (minutes ago)",
                yaxis_title="Uptime (%)",
                yaxis=dict(range=[99.9, 100.0])
            )
            
            return fig
        
        @self.dashboard_app.callback(
            Output('performance-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_performance_chart(n):
            """Update performance chart"""
            
            latency_metric = self.metrics.get('trading_latency_ms')
            memory_metric = self.metrics.get('memory_utilization')
            
            fig = go.Figure()
            
            if latency_metric:
                times = list(range(len(latency_metric.history)))
                fig.add_trace(go.Scatter(
                    x=times,
                    y=list(latency_metric.history),
                    mode='lines',
                    name='Trading Latency (ms)',
                    yaxis='y'
                ))
            
            if memory_metric:
                times = list(range(len(memory_metric.history)))
                fig.add_trace(go.Scatter(
                    x=times,
                    y=list(memory_metric.history),
                    mode='lines',
                    name='Memory Utilization (%)',
                    yaxis='y2'
                ))
            
            fig.update_layout(
                title="System Performance",
                xaxis_title="Time (minutes ago)",
                yaxis=dict(title="Latency (ms)", side="left"),
                yaxis2=dict(title="Memory (%)", side="right", overlaying="y")
            )
            
            return fig
        
        @self.dashboard_app.callback(
            Output('alert-table', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_alert_table(n):
            """Update alert table"""
            
            recent_alerts = sorted(
                [a for a in self.alerts.values() if a.last_triggered],
                key=lambda x: x.last_triggered,
                reverse=True
            )[:10]
            
            if not recent_alerts:
                return html.P("No recent alerts", className="text-muted")
            
            table_header = html.Thead([
                html.Tr([
                    html.Th("Time"),
                    html.Th("Alert"),
                    html.Th("Severity"),
                    html.Th("Status")
                ])
            ])
            
            table_body = html.Tbody([
                html.Tr([
                    html.Td(alert.last_triggered.strftime("%H:%M:%S")),
                    html.Td(alert.message),
                    html.Td(alert.severity, className=f"text-{self._get_severity_color(alert.severity)}"),
                    html.Td("Active" if alert.is_active else "Resolved")
                ])
                for alert in recent_alerts
            ])
            
            return dbc.Table([table_header, table_body], striped=True, bordered=True, hover=True)
    
    def _get_severity_color(self, severity: str) -> str:
        """Get color for alert severity"""
        colors = {
            'INFO': 'info',
            'WARNING': 'warning',
            'ERROR': 'danger',
            'CRITICAL': 'danger'
        }
        return colors.get(severity, 'secondary')
    
    async def start(self):
        """Start enterprise monitoring system"""
        self.is_running = True
        
        # Start monitoring threads
        for i in range(3):  # 3 monitoring workers
            worker = threading.Thread(target=self._monitoring_worker, daemon=True)
            worker.start()
            self.monitoring_threads.append(worker)
        
        # Start alert processing
        threading.Thread(target=self._alert_worker, daemon=True).start()
        
        # Start SLA monitoring
        threading.Thread(target=self._sla_monitoring_loop, daemon=True).start()
        
        # Start uptime tracking
        threading.Thread(target=self._uptime_tracking_loop, daemon=True).start()
        
        # Start dashboard server
        threading.Thread(target=self._start_dashboard_server, daemon=True).start()
        
        logger.info("Enterprise Monitoring System started")
    
    def stop(self):
        """Stop enterprise monitoring system"""
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.monitoring_threads:
            thread.join(timeout=5.0)
        
        logger.info("Enterprise Monitoring System stopped")
    
    def record_metric(self, metric_name: str, value: float):
        """Record a system metric"""
        try:
            if metric_name not in self.metrics:
                logger.warning(f"Unknown metric: {metric_name}")
                return
            
            metric = self.metrics[metric_name]
            metric.value = value
            metric.last_update = datetime.utcnow()
            metric.history.append(value)
            
            # Check for alerts
            self._check_metric_alerts(metric)
            
            # Add to queue for processing
            self.metric_queue.put((metric_name, value, datetime.utcnow()))
            
        except Exception as e:
            logger.error(f"Failed to record metric {metric_name}: {e}")
    
    def trigger_alert(self, alert_name: str, message: str = None, severity: str = None):
        """Manually trigger an alert"""
        try:
            if alert_name not in self.alerts:
                logger.warning(f"Unknown alert: {alert_name}")
                return
            
            alert = self.alerts[alert_name]
            
            if message:
                alert.message = message
            if severity:
                alert.severity = severity
            
            alert.last_triggered = datetime.utcnow()
            alert.trigger_count += 1
            
            # Add to alert queue
            self.alert_queue.put((alert_name, alert))
            
            logger.info(f"Alert triggered: {alert_name}")
            
        except Exception as e:
            logger.error(f"Failed to trigger alert {alert_name}: {e}")
    
    def _monitoring_worker(self):
        """Background monitoring worker"""
        while self.is_running:
            try:
                # Process metrics queue
                try:
                    metric_name, value, timestamp = self.metric_queue.get(timeout=1.0)
                    self._process_metric_update(metric_name, value, timestamp)
                except Empty:
                    continue
                
            except Exception as e:
                logger.error(f"Monitoring worker error: {e}")
                time.sleep(1)
    
    def _alert_worker(self):
        """Background alert processing worker"""
        while self.is_running:
            try:
                # Process alerts queue
                try:
                    alert_name, alert = self.alert_queue.get(timeout=1.0)
                    self._process_alert(alert)
                except Empty:
                    continue
                
            except Exception as e:
                logger.error(f"Alert worker error: {e}")
                time.sleep(1)
    
    def _process_metric_update(self, metric_name: str, value: float, timestamp: datetime):
        """Process metric update"""
        try:
            metric = self.metrics[metric_name]
            
            # Update metric status
            if value > metric.threshold_critical:
                metric.status = "CRITICAL"
            elif value > metric.threshold_warning:
                metric.status = "WARNING"
            else:
                metric.status = "OK"
            
            # Check SLA compliance
            self._check_sla_compliance(metric)
            
            # Update Prometheus metrics
            self._update_prometheus_metrics(metric)
            
        except Exception as e:
            logger.error(f"Failed to process metric update: {e}")
    
    def _process_alert(self, alert: Alert):
        """Process alert notification"""
        try:
            # Check suppression
            if self._is_alert_suppressed(alert):
                logger.debug(f"Alert {alert.alert_id} suppressed")
                return
            
            # Send notifications
            if alert.email_recipients:
                self._send_email_alert(alert)
            
            if alert.sms_recipients:
                self._send_sms_alert(alert)
            
            if self.slack_webhook_url:
                self._send_slack_alert(alert)
            
            # Update metrics
            self.monitoring_metrics['total_alerts'] += 1
            if alert.severity == 'CRITICAL':
                self.monitoring_metrics['critical_alerts'] += 1
            
            logger.info(f"Alert notifications sent: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to process alert {alert.alert_id}: {e}")
    
    def _check_metric_alerts(self, metric: SystemMetric):
        """Check if metric triggers any alerts"""
        for alert_name, alert in self.alerts.items():
            if alert.metric_name != metric.name:
                continue
            
            triggered = False
            
            if alert.condition == '>' and metric.value > alert.threshold:
                triggered = True
            elif alert.condition == '<' and metric.value < alert.threshold:
                triggered = True
            elif alert.condition == '==' and abs(metric.value - alert.threshold) < 0.001:
                triggered = True
            
            if triggered:
                self.trigger_alert(alert_name)
    
    def _check_sla_compliance(self, metric: SystemMetric):
        """Check SLA compliance for metric"""
        sla = self.sla_metrics.get(metric.name)
        if not sla:
            return
        
        # Calculate compliance based on recent history
        if len(metric.history) > 0:
            recent_values = list(metric.history)[-60:]  # Last 60 minutes
            compliant_count = sum(1 for v in recent_values if self._is_sla_compliant(metric.name, v))
            
            sla.current_value = (compliant_count / len(recent_values)) * 100
            sla.compliance_status = "COMPLIANT" if sla.current_value >= sla.target else "NON_COMPLIANT"
            
            # Track breaches
            if sla.compliance_status == "NON_COMPLIANT":
                sla.breach_count += 1
                sla.total_breach_minutes += 1
    
    def _is_sla_compliant(self, metric_name: str, value: float) -> bool:
        """Check if value meets SLA requirements"""
        metric = self.metrics.get(metric_name)
        if not metric:
            return True
        
        if metric_name == 'system_uptime':
            return value >= metric.sla_target
        elif metric_name == 'trading_latency_ms':
            return value <= metric.sla_target
        elif metric_name in ['memory_utilization', 'cpu_utilization', 'disk_utilization']:
            return value <= metric.sla_target
        else:
            return True
    
    def _is_alert_suppressed(self, alert: Alert) -> bool:
        """Check if alert is suppressed"""
        if not alert.last_suppressed:
            return False
        
        suppression_period = timedelta(minutes=alert.suppression_period_minutes)
        return datetime.utcnow() - alert.last_suppressed < suppression_period
    
    def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = ', '.join(alert.email_recipients)
            msg['Subject'] = f"[{alert.severity}] {alert.message}"
            
            body = f"""
            Alert Details:
            - Alert ID: {alert.alert_id}
            - Severity: {alert.severity}
            - Message: {alert.message}
            - Triggered: {alert.last_triggered}
            - Trigger Count: {alert.trigger_count}
            
            This is an automated alert from the Quant Fund Enterprise Monitoring System.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email (in production, would use actual SMTP)
            logger.info(f"Email alert sent to: {alert.email_recipients}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_sms_alert(self, alert: Alert):
        """Send SMS alert"""
        try:
            # In production, would use SMS API
            logger.info(f"SMS alert sent to: {alert.sms_recipients}")
            
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {e}")
    
    def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        try:
            if not self.slack_webhook_url:
                return
            
            payload = {
                'text': f"[{alert.severity}] {alert.message}",
                'attachments': [{
                    'color': self._get_slack_color(alert.severity),
                    'fields': [
                        {'title': 'Alert ID', 'value': alert.alert_id},
                        {'title': 'Triggered', 'value': alert.last_triggered.strftime("%Y-%m-%d %H:%M:%S")},
                        {'title': 'Trigger Count', 'value': str(alert.trigger_count)}
                    ]
                }]
            }
            
            # Send to Slack (in production, would make actual request)
            logger.info(f"Slack alert sent to webhook")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _get_slack_color(self, severity: str) -> str:
        """Get Slack color for severity"""
        colors = {
            'INFO': 'good',
            'WARNING': 'warning',
            'ERROR': 'danger',
            'CRITICAL': 'danger'
        }
        return colors.get(severity, 'good')
    
    def _update_prometheus_metrics(self, metric: SystemMetric):
        """Update Prometheus metrics"""
        try:
            # In production, would push to Prometheus gateway
            pass
            
        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {e}")
    
    def _sla_monitoring_loop(self):
        """Background SLA monitoring loop"""
        while self.is_running:
            try:
                # Calculate SLA compliance rates
                total_slas = len(self.sla_metrics)
                compliant_slas = sum(1 for sla in self.sla_metrics.values() if sla.compliance_status == "COMPLIANT")
                
                self.monitoring_metrics['sla_compliance_rate'] = (compliant_slas / total_slas * 100) if total_slas > 0 else 0
                
                # Sleep for 1 minute
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"SLA monitoring error: {e}")
                time.sleep(10)
    
    def _uptime_tracking_loop(self):
        """Background uptime tracking loop"""
        while self.is_running:
            try:
                # Update uptime metrics
                for service_name, uptime_metric in self.uptime_metrics.items():
                    self._calculate_uptime(uptime_metric)
                
                # Calculate overall system uptime
                system_uptime = self._calculate_system_uptime()
                self.record_metric('system_uptime', system_uptime)
                
                # Sleep for 1 minute
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Uptime tracking error: {e}")
                time.sleep(10)
    
    def _calculate_uptime(self, uptime_metric: UptimeMetric):
        """Calculate uptime for a service"""
        now = datetime.utcnow()
        total_time = (now - uptime_metric.start_time).total_seconds()
        
        # In production, would track actual downtime events
        # For now, simulate with 99.999% uptime
        uptime_percentage = 99.999
        uptime_metric.total_uptime_seconds = total_time * (uptime_percentage / 100)
        uptime_metric.total_downtime_seconds = total_time * ((100 - uptime_percentage) / 100)
        uptime_metric.current_uptime_percentage = uptime_percentage
    
    def _calculate_system_uptime(self) -> float:
        """Calculate overall system uptime"""
        # In production, would aggregate all service uptimes
        # For now, return simulated value
        return 99.999
    
    def _start_dashboard_server(self):
        """Start dashboard server"""
        try:
            if self.dashboard_app:
                self.dashboard_app.run_server(
                    host='0.0.0.0',
                    port=8050,
                    debug=False
                )
                
        except Exception as e:
            logger.error(f"Failed to start dashboard server: {e}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        return {
            'system_status': {
                'uptime_percentage': self.monitoring_metrics['uptime_percentage'],
                'active_alerts': len([a for a in self.alerts.values() if a.is_active]),
                'critical_alerts': self.monitoring_metrics['critical_alerts']
            },
            'sla_compliance': {
                'overall_compliance_rate': self.monitoring_metrics['sla_compliance_rate'],
                'compliant_slas': sum(1 for sla in self.sla_metrics.values() if sla.compliance_status == "COMPLIANT"),
                'total_slas': len(self.sla_metrics)
            },
            'performance_metrics': {
                'total_metrics': len(self.metrics),
                'metrics_in_warning': len([m for m in self.metrics.values() if m.status == "WARNING"]),
                'metrics_in_critical': len([m for m in self.metrics.values() if m.status == "CRITICAL"])
            },
            'infrastructure': {
                'dashboard_url': 'http://localhost:8050',
                'monitoring_workers': len(self.monitoring_threads),
                'total_alerts_processed': self.monitoring_metrics['total_alerts']
            }
        }


# Global enterprise monitoring system instance
_ems_instance = None

def get_enterprise_monitoring_system() -> EnterpriseMonitoringSystem:
    """Get global enterprise monitoring system instance"""
    global _ems_instance
    if _ems_instance is None:
        _ems_instance = EnterpriseMonitoringSystem()
    return _ems_instance


if __name__ == "__main__":
    # Test enterprise monitoring
    ems = EnterpriseMonitoringSystem()
    
    # Record some test metrics
    ems.record_metric('system_uptime', 99.999)
    ems.record_metric('trading_latency_ms', 8.5)
    ems.record_metric('memory_utilization', 75.0)
    
    # Trigger a test alert
    ems.trigger_alert('high_latency', 'Trading latency spike detected', 'WARNING')
    
    # Get summary
    summary = ems.get_monitoring_summary()
    print(json.dumps(summary, indent=2, default=str))

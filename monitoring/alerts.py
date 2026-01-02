# monitoring/alerts.py
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)

class AlertManager:
    """
    Monitor system metrics and send alerts when thresholds are breached.
    
    Setup for email alerts (optional, free with Gmail):
    - ALERT_EMAIL: Your Gmail address
    - ALERT_PASSWORD: Gmail app password (not your regular password!)
    - ALERT_TO: Recipient email
    
    Generate Gmail app password: https://myaccount.google.com/apppasswords
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.email_enabled = all([
            os.getenv("ALERT_EMAIL"),
            os.getenv("ALERT_PASSWORD"),
            os.getenv("ALERT_TO")
        ])
        
        if not self.email_enabled:
            logger.info("Email alerts disabled (env vars not set)")
    
    def _default_config(self) -> Dict:
        """Default alert thresholds."""
        return {
            "max_drawdown_pct": 0.10,  # Alert if DD > 10%
            "volatility_spike_factor": 3.0,  # Alert if vol > 3x normal
            "large_loss_pct": 0.05,  # Alert if single-day loss > 5%
        }
    
    def check_drawdown(self, current_dd: float):
        """Alert if drawdown exceeds threshold."""
        if abs(current_dd) > self.config["max_drawdown_pct"]:
            self._send_alert(
                subject="⚠️ Max Drawdown Alert",
                message=f"Drawdown: {current_dd*100:.2f}% (Threshold: {self.config['max_drawdown_pct']*100:.0f}%)"
            )
    
    def check_volatility(self, current_vol: float, baseline_vol: float):
        """Alert if volatility spikes."""
        if current_vol > baseline_vol * self.config["volatility_spike_factor"]:
            self._send_alert(
                subject="⚠️ Volatility Spike Alert",
                message=f"Current Vol: {current_vol*100:.2f}% (Baseline: {baseline_vol*100:.2f}%)"
            )
    
    def check_daily_loss(self, loss_pct: float):
        """Alert if single-day loss is large."""
        if abs(loss_pct) > self.config["large_loss_pct"]:
            self._send_alert(
                subject="⚠️ Large Loss Alert",
                message=f"Daily Loss: {loss_pct*100:.2f}% (Threshold: {self.config['large_loss_pct']*100:.0f}%)"
            )
    
    def _send_alert(self, subject: str, message: str):
        """Send alert via email and log."""
        logger.warning(f"ALERT: {subject} - {message}")
        
        if self.email_enabled:
            try:
                self._send_email(subject, message)
            except Exception as e:
                logger.error(f"Failed to send email alert: {e}")
    
    def _send_email(self, subject: str, body: str):
        """Send email via Gmail SMTP."""
        sender_email = os.getenv("ALERT_EMAIL")
        sender_password = os.getenv("ALERT_PASSWORD")
        recipient_email = os.getenv("ALERT_TO")
        
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = f"[Mini Quant Fund] {subject}"
        
        msg.attach(MIMEText(body, "plain"))
        
        # Gmail SMTP
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        logger.info(f"Email alert sent to {recipient_email}")

"""
Security Infrastructure - Production Implementation
Enterprise-grade security for trading system
"""

import asyncio
import logging
import json
import hashlib
import hmac
import base64
import os
import secrets
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import ssl
import bcrypt
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import aiohttp
import sqlite3
import uuid
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"

class UserRole(Enum):
    """User roles"""
    TRADER = "trader"
    RISK_MANAGER = "risk_manager"
    COMPLIANCE_OFFICER = "compliance_officer"
    SYSTEM_ADMIN = "system_admin"
    VIEWER = "viewer"

class AuthenticationMethod(Enum):
    """Authentication methods"""
    PASSWORD = "password"
    MFA = "mfa"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"

@dataclass
class User:
    """User structure"""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime]
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    api_keys: List[str] = field(default_factory=list)
    certificate_fingerprint: Optional[str] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None

@dataclass
class SecurityEvent:
    """Security event structure"""
    event_id: str
    event_type: str
    severity: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    timestamp: datetime
    description: str
    details: Dict[str, Any]
    resolved: bool = False

@dataclass
class APIKey:
    """API key structure"""
    key_id: str
    key_hash: str
    user_id: str
    permissions: List[str]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    is_active: bool = True
    rate_limit: int = 1000  # requests per hour

class SecurityInfrastructure:
    """Production security infrastructure"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.users = {}
        self.api_keys = {}
        self.security_events = {}
        self.active_sessions = {}
        self.encryption_keys = {}
        self.running = False
        self.db_connection = None
        self.rate_limits = {}
        self.failed_attempts = {}
        
        # Initialize security components
        self._initialize_encryption()
        self._initialize_database()
        self._initialize_ssl_context()
        
    def _initialize_encryption(self):
        """Initialize encryption keys"""
        # Generate master key
        self.master_key = self._generate_master_key()
        
        # Generate data encryption key
        self.data_key = Fernet.generate_key()
        
        # Generate JWT signing key
        self.jwt_secret = secrets.token_urlsafe(32)
        
        # Store keys
        self.encryption_keys = {
            'master': self.master_key,
            'data': self.data_key,
            'jwt': self.jwt_secret
        }
    
    def _generate_master_key(self) -> bytes:
        """Generate master encryption key"""
        password = self.config.get('master_password', 'default_master_password').encode()
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def _initialize_database(self):
        """Initialize security database"""
        try:
            self.db_connection = sqlite3.connect('security.db')
            cursor = self.db_connection.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE,
                    email TEXT UNIQUE,
                    role TEXT,
                    permissions TEXT,
                    password_hash TEXT,
                    mfa_enabled BOOLEAN,
                    mfa_secret TEXT,
                    api_keys TEXT,
                    certificate_fingerprint TEXT,
                    is_active BOOLEAN,
                    created_at TEXT,
                    last_login TEXT,
                    failed_login_attempts INTEGER,
                    locked_until TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    key_hash TEXT UNIQUE,
                    user_id TEXT,
                    permissions TEXT,
                    created_at TEXT,
                    expires_at TEXT,
                    last_used TEXT,
                    is_active BOOLEAN,
                    rate_limit INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT,
                    severity TEXT,
                    user_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TEXT,
                    description TEXT,
                    details TEXT,
                    resolved BOOLEAN
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TEXT,
                    expires_at TEXT,
                    is_active BOOLEAN
                )
            ''')
            
            self.db_connection.commit()
            logger.info("Security database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize security database: {e}")
            raise
    
    def _initialize_ssl_context(self):
        """Initialize SSL context"""
        self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        self.ssl_context.check_hostname = True
        self.ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        # Load certificates if available
        cert_file = self.config.get('ssl_cert_file')
        key_file = self.config.get('ssl_key_file')
        
        if cert_file and key_file and os.path.exists(cert_file) and os.path.exists(key_file):
            self.ssl_context.load_cert_chain(cert_file, key_file)
            logger.info("SSL certificates loaded")
        else:
            logger.warning("SSL certificates not found, using default context")
    
    async def start(self):
        """Start security infrastructure"""
        self.running = True
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_security_events())
        asyncio.create_task(self._monitor_failed_logins())
        asyncio.create_task(self._cleanup_expired_sessions())
        asyncio.create_task(self._monitor_rate_limits())
        asyncio.create_task(self._periodic_security_scan())
        
        logger.info("Security infrastructure started")
    
    async def stop(self):
        """Stop security infrastructure"""
        self.running = False
        
        if self.db_connection:
            self.db_connection.close()
        
        logger.info("Security infrastructure stopped")
    
    async def create_user(self, username: str, email: str, role: UserRole, 
                         password: str, permissions: List[str]) -> Optional[User]:
        """Create new user"""
        try:
            # Check if user already exists
            if self._user_exists(username, email):
                return None
            
            # Hash password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            # Generate user ID
            user_id = str(uuid.uuid4())
            
            # Create user
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                role=role,
                permissions=permissions,
                created_at=datetime.utcnow()
            )
            
            # Store in database
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO users (
                    user_id, username, email, role, permissions, password_hash,
                    mfa_enabled, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, username, email, role.value, json.dumps(permissions),
                password_hash.decode('utf-8'), False, user.created_at.isoformat()
            ))
            
            self.db_connection.commit()
            
            # Store in memory
            self.users[user_id] = user
            
            # Log security event
            await self._log_security_event(
                "USER_CREATED",
                "INFO",
                user_id,
                "127.0.0.1",
                "System",
                f"User {username} created with role {role.value}",
                {"username": username, "role": role.value}
            )
            
            logger.info(f"User created: {username}")
            return user
            
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return None
    
    def _user_exists(self, username: str, email: str) -> bool:
        """Check if user already exists"""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM users WHERE username = ? OR email = ?
        ''', (username, email))
        
        return cursor.fetchone()[0] > 0
    
    async def authenticate_user(self, username: str, password: str, 
                              ip_address: str, user_agent: str) -> Optional[str]:
        """Authenticate user and return session token"""
        try:
            # Get user
            cursor = self.db_connection.cursor()
            cursor.execute('''
                SELECT user_id, username, password_hash, is_active, failed_login_attempts, locked_until
                FROM users WHERE username = ?
            ''', (username,))
            
            user_data = cursor.fetchone()
            
            if not user_data:
                await self._log_security_event(
                    "LOGIN_FAILED",
                    "WARNING",
                    None,
                    ip_address,
                    user_agent,
                    f"Login failed for unknown user: {username}",
                    {"username": username}
                )
                return None
            
            user_id, username, password_hash, is_active, failed_attempts, locked_until = user_data
            
            # Check if user is locked
            if locked_until:
                locked_until_dt = datetime.fromisoformat(locked_until)
                if datetime.utcnow() < locked_until_dt:
                    await self._log_security_event(
                        "LOGIN_BLOCKED",
                        "WARNING",
                        user_id,
                        ip_address,
                        user_agent,
                        f"Login blocked for locked user: {username}",
                        {"username": username, "locked_until": locked_until}
                    )
                    return None
                else:
                    # Unlock user
                    cursor.execute('''
                        UPDATE users SET failed_login_attempts = 0, locked_until = NULL
                        WHERE user_id = ?
                    ''', (user_id,))
                    self.db_connection.commit()
            
            # Check if user is active
            if not is_active:
                await self._log_security_event(
                    "LOGIN_FAILED",
                    "WARNING",
                    user_id,
                    ip_address,
                    user_agent,
                    f"Login failed for inactive user: {username}",
                    {"username": username}
                )
                return None
            
            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
                # Increment failed attempts
                failed_attempts += 1
                cursor.execute('''
                    UPDATE users SET failed_login_attempts = ?
                    WHERE user_id = ?
                ''', (failed_attempts, user_id))
                self.db_connection.commit()
                
                # Lock user if too many failed attempts
                if failed_attempts >= 5:
                    locked_until = datetime.utcnow() + timedelta(minutes=30)
                    cursor.execute('''
                        UPDATE users SET locked_until = ?
                        WHERE user_id = ?
                    ''', (locked_until.isoformat(), user_id))
                    self.db_connection.commit()
                
                await self._log_security_event(
                    "LOGIN_FAILED",
                    "WARNING",
                    user_id,
                    ip_address,
                    user_agent,
                    f"Invalid password for user: {username}",
                    {"username": username, "failed_attempts": failed_attempts}
                )
                return None
            
            # Reset failed attempts
            cursor.execute('''
                UPDATE users SET failed_login_attempts = 0, last_login = ?
                WHERE user_id = ?
            ''', (datetime.utcnow().isoformat(), user_id))
            self.db_connection.commit()
            
            # Create session
            session_id = await self._create_session(user_id, ip_address, user_agent)
            
            await self._log_security_event(
                "LOGIN_SUCCESS",
                "INFO",
                user_id,
                ip_address,
                user_agent,
                f"User logged in: {username}",
                {"username": username}
            )
            
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to authenticate user: {e}")
            return None
    
    async def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """Create user session"""
        session_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=8)
        
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO sessions (session_id, user_id, ip_address, user_agent, created_at, expires_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, user_id, ip_address, user_agent, datetime.utcnow().isoformat(), expires_at.isoformat(), True))
        
        self.db_connection.commit()
        
        # Store in memory
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'created_at': datetime.utcnow(),
            'expires_at': expires_at
        }
        
        return session_id
    
    async def validate_session(self, session_id: str, ip_address: str) -> Optional[User]:
        """Validate session and return user"""
        try:
            # Check if session exists and is active
            cursor = self.db_connection.cursor()
            cursor.execute('''
                SELECT user_id, ip_address, expires_at, is_active FROM sessions
                WHERE session_id = ?
            ''', (session_id,))
            
            session_data = cursor.fetchone()
            
            if not session_data:
                return None
            
            user_id, session_ip, expires_at, is_active = session_data
            
            # Check if session is active
            if not is_active:
                return None
            
            # Check if session has expired
            expires_at_dt = datetime.fromisoformat(expires_at)
            if datetime.utcnow() > expires_at_dt:
                # Deactivate session
                cursor.execute('''
                    UPDATE sessions SET is_active = 0 WHERE session_id = ?
                ''', (session_id,))
                self.db_connection.commit()
                return None
            
            # Check IP address (optional security measure)
            if session_ip != ip_address:
                await self._log_security_event(
                    "SESSION_IP_MISMATCH",
                    "WARNING",
                    user_id,
                    ip_address,
                    "Unknown",
                    f"Session IP mismatch: {session_ip} vs {ip_address}",
                    {"session_id": session_id, "expected_ip": session_ip, "actual_ip": ip_address}
                )
                # Could choose to invalidate session here
                return None
            
            # Get user
            cursor.execute('''
                SELECT user_id, username, email, role, permissions, is_active FROM users
                WHERE user_id = ?
            ''', (user_id,))
            
            user_data = cursor.fetchone()
            
            if not user_data:
                return None
            
            user_id, username, email, role, permissions, is_active = user_data
            
            if not is_active:
                return None
            
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                role=UserRole(role),
                permissions=json.loads(permissions),
                created_at=datetime.utcnow()
            )
            
            return user
            
        except Exception as e:
            logger.error(f"Failed to validate session: {e}")
            return None
    
    async def create_api_key(self, user_id: str, permissions: List[str], 
                           expires_in_days: Optional[int] = None) -> Optional[str]:
        """Create API key for user"""
        try:
            # Generate API key
            api_key = secrets.token_urlsafe(32)
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Calculate expiration
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            # Store in database
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO api_keys (key_id, key_hash, user_id, permissions, created_at, expires_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (str(uuid.uuid4()), key_hash, user_id, json.dumps(permissions), 
                  datetime.utcnow().isoformat(), expires_at.isoformat() if expires_at else None, True))
            
            self.db_connection.commit()
            
            # Store in memory
            self.api_keys[key_hash] = {
                'user_id': user_id,
                'permissions': permissions,
                'created_at': datetime.utcnow(),
                'expires_at': expires_at,
                'is_active': True
            }
            
            await self._log_security_event(
                "API_KEY_CREATED",
                "INFO",
                user_id,
                "127.0.0.1",
                "System",
                f"API key created for user {user_id}",
                {"user_id": user_id, "permissions": permissions}
            )
            
            return api_key
            
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            return None
    
    async def validate_api_key(self, api_key: str, ip_address: str) -> Optional[User]:
        """Validate API key and return user"""
        try:
            # Hash API key
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Check if key exists
            cursor = self.db_connection.cursor()
            cursor.execute('''
                SELECT user_id, permissions, expires_at, is_active, rate_limit FROM api_keys
                WHERE key_hash = ?
            ''', (key_hash,))
            
            key_data = cursor.fetchone()
            
            if not key_data:
                return None
            
            user_id, permissions, expires_at, is_active, rate_limit = key_data
            
            # Check if key is active
            if not is_active:
                return None
            
            # Check if key has expired
            if expires_at:
                expires_at_dt = datetime.fromisoformat(expires_at)
                if datetime.utcnow() > expires_at_dt:
                    # Deactivate key
                    cursor.execute('''
                        UPDATE api_keys SET is_active = 0 WHERE key_hash = ?
                    ''', (key_hash,))
                    self.db_connection.commit()
                    return None
            
            # Check rate limit
            if not await self._check_rate_limit(key_hash, rate_limit, ip_address):
                await self._log_security_event(
                    "RATE_LIMIT_EXCEEDED",
                    "WARNING",
                    user_id,
                    ip_address,
                    "Unknown",
                    f"Rate limit exceeded for API key",
                    {"key_hash": key_hash, "rate_limit": rate_limit}
                )
                return None
            
            # Update last used
            cursor.execute('''
                UPDATE api_keys SET last_used = ? WHERE key_hash = ?
            ''', (datetime.utcnow().isoformat(), key_hash))
            self.db_connection.commit()
            
            # Get user
            cursor.execute('''
                SELECT user_id, username, email, role, permissions, is_active FROM users
                WHERE user_id = ?
            ''', (user_id,))
            
            user_data = cursor.fetchone()
            
            if not user_data:
                return None
            
            user_id, username, email, role, user_permissions, is_active = user_data
            
            if not is_active:
                return None
            
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                role=UserRole(role),
                permissions=json.loads(user_permissions),
                created_at=datetime.utcnow()
            )
            
            return user
            
        except Exception as e:
            logger.error(f"Failed to validate API key: {e}")
            return None
    
    async def _check_rate_limit(self, key_hash: str, rate_limit: int, ip_address: str) -> bool:
        """Check rate limit for API key"""
        current_time = datetime.utcnow()
        hour_start = current_time.replace(minute=0, second=0, microsecond=0)
        
        # Get current usage
        if key_hash not in self.rate_limits:
            self.rate_limits[key_hash] = {}
        
        if hour_start not in self.rate_limits[key_hash]:
            self.rate_limits[key_hash][hour_start] = 0
        
        # Check if under limit
        if self.rate_limits[key_hash][hour_start] >= rate_limit:
            return False
        
        # Increment usage
        self.rate_limits[key_hash][hour_start] += 1
        
        return True
    
    async def encrypt_data(self, data: str, security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL) -> str:
        """Encrypt data"""
        try:
            if security_level == SecurityLevel.PUBLIC:
                return data
            
            fernet = Fernet(self.data_key)
            encrypted_data = fernet.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
            
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            return data
    
    async def decrypt_data(self, encrypted_data: str, security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL) -> str:
        """Decrypt data"""
        try:
            if security_level == SecurityLevel.PUBLIC:
                return encrypted_data
            
            fernet = Fernet(self.data_key)
            decoded_data = base64.b64decode(encrypted_data.encode())
            decrypted_data = fernet.decrypt(decoded_data)
            return decrypted_data.decode()
            
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            return encrypted_data
    
    async def _log_security_event(self, event_type: str, severity: str, user_id: Optional[str],
                                ip_address: str, user_agent: str, description: str, 
                                details: Dict[str, Any]):
        """Log security event"""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow(),
            description=description,
            details=details
        )
        
        self.security_events[event.event_id] = event
        
        # Store in database
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO security_events (
                event_id, event_type, severity, user_id, ip_address, user_agent,
                timestamp, description, details, resolved
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id, event.event_type, event.severity, event.user_id,
            event.ip_address, event.user_agent, event.timestamp.isoformat(),
            event.description, json.dumps(event.details), event.resolved
        ))
        
        self.db_connection.commit()
        
        # Log to file
        logger.info(f"Security Event: {event_type} - {description}")
    
    async def _monitor_security_events(self):
        """Monitor security events"""
        while self.running:
            try:
                # Check for critical events
                current_time = datetime.utcnow()
                recent_events = [
                    event for event in self.security_events.values()
                    if event.timestamp > current_time - timedelta(minutes=5)
                    and event.severity in ['CRITICAL', 'ERROR']
                ]
                
                if recent_events:
                    logger.warning(f"Critical security events detected: {len(recent_events)}")
                    # Could trigger alerts, notifications, etc.
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring security events: {e}")
                await asyncio.sleep(30)
    
    async def _monitor_failed_logins(self):
        """Monitor failed login attempts"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                recent_failed_logins = [
                    event for event in self.security_events.values()
                    if event.timestamp > current_time - timedelta(minutes=10)
                    and event.event_type == 'LOGIN_FAILED'
                    and event.ip_address
                ]
                
                # Group by IP address
                ip_failures = {}
                for event in recent_failed_logins:
                    ip = event.ip_address
                    if ip not in ip_failures:
                        ip_failures[ip] = 0
                    ip_failures[ip] += 1
                
                # Check for suspicious activity
                for ip, failures in ip_failures.items():
                    if failures >= 10:
                        await self._log_security_event(
                            "SUSPICIOUS_LOGIN_ACTIVITY",
                            "WARNING",
                            None,
                            ip,
                            "Unknown",
                            f"Suspicious login activity from IP: {failures} failures",
                            {"ip_address": ip, "failures": failures}
                        )
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring failed logins: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                cursor = self.db_connection.cursor()
                cursor.execute('''
                    SELECT session_id FROM sessions WHERE expires_at < ? AND is_active = 1
                ''', (current_time.isoformat(),))
                
                expired_sessions = cursor.fetchall()
                
                for session in expired_sessions:
                    session_id = session[0]
                    cursor.execute('''
                        UPDATE sessions SET is_active = 0 WHERE session_id = ?
                    ''', (session_id,))
                    
                    # Remove from memory
                    if session_id in self.active_sessions:
                        del self.active_sessions[session_id]
                
                self.db_connection.commit()
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up expired sessions: {e}")
                await asyncio.sleep(300)
    
    async def _monitor_rate_limits(self):
        """Monitor rate limits"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                hour_start = current_time.replace(minute=0, second=0, microsecond=0)
                
                # Clean old rate limit data
                for key_hash in list(self.rate_limits.keys()):
                    for hour in list(self.rate_limits[key_hash].keys()):
                        if hour < hour_start - timedelta(hours=24):
                            del self.rate_limits[key_hash][hour]
                    
                    if not self.rate_limits[key_hash]:
                        del self.rate_limits[key_hash]
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring rate limits: {e}")
                await asyncio.sleep(60)
    
    async def _periodic_security_scan(self):
        """Perform periodic security scans"""
        while self.running:
            try:
                # Scan for inactive users with active sessions
                cursor = self.db_connection.cursor()
                cursor.execute('''
                    SELECT u.user_id, u.username, s.session_id FROM users u
                    JOIN sessions s ON u.user_id = s.user_id
                    WHERE u.is_active = 0 AND s.is_active = 1
                ''')
                
                inactive_user_sessions = cursor.fetchall()
                
                for session in inactive_user_sessions:
                    user_id, username, session_id = session
                    cursor.execute('''
                        UPDATE sessions SET is_active = 0 WHERE session_id = ?
                    ''', (session_id,))
                    
                    await self._log_security_event(
                        "INACTIVE_USER_SESSION",
                        "WARNING",
                        user_id,
                        "127.0.0.1",
                        "System",
                        f"Inactive user {username} had active session",
                        {"user_id": user_id, "username": username, "session_id": session_id}
                    )
                
                self.db_connection.commit()
                
                await asyncio.sleep(3600)  # Scan every hour
                
            except Exception as e:
                logger.error(f"Error in periodic security scan: {e}")
                await asyncio.sleep(300)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary"""
        cursor = self.db_connection.cursor()
        
        # Get user counts
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
        active_users = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 0')
        inactive_users = cursor.fetchone()[0]
        
        # Get session counts
        cursor.execute('SELECT COUNT(*) FROM sessions WHERE is_active = 1')
        active_sessions = cursor.fetchone()[0]
        
        # Get API key counts
        cursor.execute('SELECT COUNT(*) FROM api_keys WHERE is_active = 1')
        active_api_keys = cursor.fetchone()[0]
        
        # Get security event counts
        cursor.execute('''
            SELECT severity, COUNT(*) as count FROM security_events
            WHERE timestamp > ?
            GROUP BY severity
        ''', ((datetime.utcnow() - timedelta(days=1)).isoformat(),))
        
        events_by_severity = dict(cursor.fetchall())
        
        return {
            'active_users': active_users,
            'inactive_users': inactive_users,
            'active_sessions': active_sessions,
            'active_api_keys': active_api_keys,
            'events_by_severity': events_by_severity,
            'total_security_events': len(self.security_events),
            'unresolved_events': len([e for e in self.security_events.values() if not e.resolved])
        }
    
    @asynccontextmanager
    async def get_secure_session(self, session_id: str, ip_address: str):
        """Get secure session context"""
        user = await self.validate_session(session_id, ip_address)
        if not user:
            raise Exception("Invalid session")
        
        try:
            yield user
        finally:
            # Session cleanup if needed
            pass

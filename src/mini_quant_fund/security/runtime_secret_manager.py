#!/usr/bin/env python3
"""
RUNTIME SECRET MANAGEMENT & ROTATION
====================================

Enterprise-grade secret management with automatic rotation,
encryption at rest, and secure distribution for MiniQuantFund.

Features:
- Automatic secret rotation with configurable intervals
- Integration with HashiCorp Vault, AWS Secrets Manager, Azure Key Vault
- In-memory encryption for cached secrets
- Audit logging of all secret access
- Graceful rotation without service interruption

Usage:
    from mini_quant_fund.security.runtime_secret_manager import SecretManager
    secrets = SecretManager()
    api_key = secrets.get_secret("alpaca_api_key")
"""

import os
import json
import time
import hashlib
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import base64

# Cryptographic imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecretSource(Enum):
    """Secret storage backends."""
    ENVIRONMENT = "environment"
    VAULT = "vault"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    AZURE_KEY_VAULT = "azure_key_vault"
    GCP_SECRET_MANAGER = "gcp_secret_manager"
    FILE = "file"
    MEMORY = "memory"


@dataclass
class SecretMetadata:
    """Metadata for a managed secret."""
    key: str
    source: SecretSource
    created_at: datetime
    expires_at: Optional[datetime]
    last_rotated: datetime
    rotation_interval: timedelta
    version: int
    checksum: str
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class SecretRotationPolicy:
    """Policy for secret rotation."""
    enabled: bool = True
    interval_days: int = 90
    emergency_rotation: bool = True
    grace_period_hours: int = 24
    notify_before_days: int = 7


class SecretManager:
    """Enterprise secret management with rotation capabilities."""
    
    def __init__(self, 
                 master_key: Optional[str] = None,
                 state_file: str = "runtime/secret_state.json",
                 auto_rotation: bool = True):
        self.master_key = master_key or os.getenv("SECRET_MASTER_KEY")
        self.state_file = Path(state_file)
        self.auto_rotation = auto_rotation
        
        self._secrets: Dict[str, Any] = {}
        self._metadata: Dict[str, SecretMetadata] = {}
        self._policies: Dict[str, SecretRotationPolicy] = {}
        self._cipher: Optional[Any] = None
        
        self._lock = threading.RLock()
        self._rotation_thread: Optional[threading.Thread] = None
        self._running = False
        
        self._initialize_encryption()
        self._load_state()
        
        if auto_rotation:
            self.start_rotation_service()
    
    def _initialize_encryption(self):
        """Initialize encryption for in-memory secrets."""
        if not CRYPTO_AVAILABLE or not self.master_key:
            logger.warning("Encryption not available - secrets will be in plaintext")
            return
        
        try:
            # Derive key from master key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"mini_quant_fund_salt",
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
            self._cipher = Fernet(key)
            logger.info("Encryption initialized")
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
    
    def _encrypt(self, plaintext: str) -> str:
        """Encrypt a string."""
        if self._cipher:
            return self._cipher.encrypt(plaintext.encode()).decode()
        return plaintext
    
    def _decrypt(self, ciphertext: str) -> str:
        """Decrypt a string."""
        if self._cipher:
            return self._cipher.decrypt(ciphertext.encode()).decode()
        return ciphertext
    
    def _load_state(self):
        """Load secret state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                
                for key, meta_dict in state.get("metadata", {}).items():
                    self._metadata[key] = SecretMetadata(
                        key=key,
                        source=SecretSource(meta_dict["source"]),
                        created_at=datetime.fromisoformat(meta_dict["created_at"]),
                        expires_at=datetime.fromisoformat(meta_dict["expires_at"]) if meta_dict.get("expires_at") else None,
                        last_rotated=datetime.fromisoformat(meta_dict["last_rotated"]),
                        rotation_interval=timedelta(days=meta_dict["rotation_interval_days"]),
                        version=meta_dict["version"],
                        checksum=meta_dict["checksum"],
                        access_count=meta_dict.get("access_count", 0),
                        last_accessed=datetime.fromisoformat(meta_dict["last_accessed"]) if meta_dict.get("last_accessed") else None
                    )
                
                logger.info(f"Loaded {len(self._metadata)} secret metadata entries")
            except Exception as e:
                logger.error(f"Failed to load secret state: {e}")
    
    def _save_state(self):
        """Save secret state to file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            state = {
                "metadata": {
                    key: {
                        "key": meta.key,
                        "source": meta.source.value,
                        "created_at": meta.created_at.isoformat(),
                        "expires_at": meta.expires_at.isoformat() if meta.expires_at else None,
                        "last_rotated": meta.last_rotated.isoformat(),
                        "rotation_interval_days": meta.rotation_interval.days,
                        "version": meta.version,
                        "checksum": meta.checksum,
                        "access_count": meta.access_count,
                        "last_accessed": meta.last_accessed.isoformat() if meta.last_accessed else None
                    }
                    for key, meta in self._metadata.items()
                },
                "saved_at": datetime.utcnow().isoformat()
            }
            
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save secret state: {e}")
    
    def register_secret(self, 
                        key: str, 
                        value: str,
                        source: SecretSource = SecretSource.MEMORY,
                        rotation_policy: Optional[SecretRotationPolicy] = None,
                        expires_in_days: Optional[int] = None):
        """Register a secret for management."""
        with self._lock:
            # Encrypt and store
            encrypted_value = self._encrypt(value)
            self._secrets[key] = encrypted_value
            
            # Create metadata
            now = datetime.utcnow()
            metadata = SecretMetadata(
                key=key,
                source=source,
                created_at=now,
                expires_at=now + timedelta(days=expires_in_days) if expires_in_days else None,
                last_rotated=now,
                rotation_interval=timedelta(days=rotation_policy.interval_days if rotation_policy else 90),
                version=1,
                checksum=hashlib.sha256(value.encode()).hexdigest()[:16]
            )
            self._metadata[key] = metadata
            
            if rotation_policy:
                self._policies[key] = rotation_policy
            
            self._save_state()
            
            logger.info(f"Registered secret: {key} (source: {source.value})")
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret value."""
        with self._lock:
            # Check if secret exists
            if key not in self._secrets:
                # Try to load from environment as fallback
                env_value = os.getenv(key.upper())
                if env_value:
                    self.register_secret(key, env_value, SecretSource.ENVIRONMENT)
                    return env_value
                return default
            
            # Decrypt and return
            encrypted = self._secrets[key]
            value = self._decrypt(encrypted)
            
            # Update access metadata
            metadata = self._metadata[key]
            metadata.access_count += 1
            metadata.last_accessed = datetime.utcnow()
            
            # Audit log
            logger.debug(f"Secret accessed: {key} (access #{metadata.access_count})")
            
            # Check expiration
            if metadata.expires_at and datetime.utcnow() > metadata.expires_at:
                logger.warning(f"Secret {key} has expired")
                if self._policies.get(key, SecretRotationPolicy()).emergency_rotation:
                    self._trigger_emergency_rotation(key)
            
            return value
    
    def rotate_secret(self, key: str, new_value: Optional[str] = None) -> bool:
        """Rotate a secret to a new value."""
        with self._lock:
            if key not in self._metadata:
                logger.error(f"Cannot rotate unknown secret: {key}")
                return False
            
            metadata = self._metadata[key]
            
            # Get new value if not provided
            if new_value is None:
                new_value = self._fetch_from_source(key, metadata.source)
                if new_value is None:
                    logger.error(f"Failed to fetch new value for {key} from {metadata.source.value}")
                    return False
            
            # Store new value
            encrypted_value = self._encrypt(new_value)
            self._secrets[key] = encrypted_value
            
            # Update metadata
            metadata.last_rotated = datetime.utcnow()
            metadata.version += 1
            metadata.checksum = hashlib.sha256(new_value.encode()).hexdigest()[:16]
            
            self._save_state()
            
            logger.info(f"Rotated secret: {key} (version {metadata.version})")
            return True
    
    def _fetch_from_source(self, key: str, source: SecretSource) -> Optional[str]:
        """Fetch secret value from configured source."""
        try:
            if source == SecretSource.ENVIRONMENT:
                return os.getenv(key.upper())
            
            elif source == SecretSource.VAULT:
                return self._fetch_from_vault(key)
            
            elif source == SecretSource.AWS_SECRETS_MANAGER:
                return self._fetch_from_aws_secrets(key)
            
            elif source == SecretSource.AZURE_KEY_VAULT:
                return self._fetch_from_azure_vault(key)
            
            elif source == SecretSource.FILE:
                file_path = Path(f"secrets/{key}")
                if file_path.exists():
                    with open(file_path, "r") as f:
                        return f.read().strip()
            
            elif source == SecretSource.MEMORY:
                # Cannot fetch from memory - return current value
                encrypted = self._secrets.get(key)
                return self._decrypt(encrypted) if encrypted else None
            
        except Exception as e:
            logger.error(f"Failed to fetch from {source.value}: {e}")
        
        return None
    
    def _fetch_from_vault(self, key: str) -> Optional[str]:
        """Fetch secret from HashiCorp Vault."""
        try:
            import hvac
            
            vault_addr = os.getenv("VAULT_ADDR", "http://localhost:8200")
            vault_token = os.getenv("VAULT_TOKEN")
            
            if not vault_token:
                return None
            
            client = hvac.Client(url=vault_addr, token=vault_token)
            secret = client.secrets.kv.v2.read_secret_version(path=f"trading/{key}")
            return secret["data"]["data"].get("value")
            
        except Exception as e:
            logger.error(f"Vault fetch failed: {e}")
            return None
    
    def _fetch_from_aws_secrets(self, key: str) -> Optional[str]:
        """Fetch secret from AWS Secrets Manager."""
        try:
            import boto3
            
            client = boto3.client("secretsmanager")
            response = client.get_secret_value(SecretId=f"mini-quant-fund/{key}")
            return response.get("SecretString")
            
        except Exception as e:
            logger.error(f"AWS Secrets Manager fetch failed: {e}")
            return None
    
    def _fetch_from_azure_vault(self, key: str) -> Optional[str]:
        """Fetch secret from Azure Key Vault."""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
            
            vault_url = os.getenv("AZURE_KEY_VAULT_URL")
            if not vault_url:
                return None
            
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=vault_url, credential=credential)
            secret = client.get_secret(key)
            return secret.value
            
        except Exception as e:
            logger.error(f"Azure Key Vault fetch failed: {e}")
            return None
    
    def _trigger_emergency_rotation(self, key: str):
        """Trigger emergency secret rotation."""
        logger.critical(f"EMERGENCY ROTATION triggered for {key}")
        
        # Notify monitoring
        try:
            from mini_quant_fund.monitoring.production_monitor import get_production_monitor
            monitor = get_production_monitor()
            monitor.alert_manager.evaluate_metrics({
                "emergency_rotation": True,
                "secret_key": key,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception:
            pass
        
        # Attempt rotation
        success = self.rotate_secret(key)
        if success:
            logger.info(f"Emergency rotation successful for {key}")
        else:
            logger.error(f"Emergency rotation failed for {key}")
    
    def start_rotation_service(self):
        """Start automatic rotation background service."""
        if self._running:
            return
        
        self._running = True
        self._rotation_thread = threading.Thread(target=self._rotation_loop, daemon=True)
        self._rotation_thread.start()
        logger.info("Secret rotation service started")
    
    def stop_rotation_service(self):
        """Stop automatic rotation service."""
        self._running = False
        if self._rotation_thread:
            self._rotation_thread.join(timeout=5.0)
        logger.info("Secret rotation service stopped")
    
    def _rotation_loop(self):
        """Background rotation loop."""
        while self._running:
            try:
                self._check_and_rotate_expired()
            except Exception as e:
                logger.error(f"Rotation loop error: {e}")
            
            # Check every hour
            time.sleep(3600)
    
    def _check_and_rotate_expired(self):
        """Check for and rotate expired secrets."""
        now = datetime.utcnow()
        
        for key, metadata in self._metadata.items():
            policy = self._policies.get(key, SecretRotationPolicy())
            
            if not policy.enabled:
                continue
            
            # Check if rotation needed
            time_since_rotation = now - metadata.last_rotated
            
            if time_since_rotation >= metadata.rotation_interval:
                logger.info(f"Scheduled rotation for {key}")
                self.rotate_secret(key)
            
            # Check expiration warning
            if metadata.expires_at:
                time_to_expiry = metadata.expires_at - now
                if time_to_expiry <= timedelta(days=policy.notify_before_days):
                    logger.warning(f"Secret {key} expires in {time_to_expiry.days} days")
    
    def get_secret_status(self, key: str) -> Optional[Dict]:
        """Get status information for a secret."""
        with self._lock:
            if key not in self._metadata:
                return None
            
            metadata = self._metadata[key]
            now = datetime.utcnow()
            
            status = {
                "key": key,
                "source": metadata.source.value,
                "version": metadata.version,
                "created_at": metadata.created_at.isoformat(),
                "last_rotated": metadata.last_rotated.isoformat(),
                "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                "access_count": metadata.access_count,
                "last_accessed": metadata.last_accessed.isoformat() if metadata.last_accessed else None,
                "days_until_rotation": (metadata.rotation_interval - (now - metadata.last_rotated)).days,
                "checksum_prefix": metadata.checksum[:8]
            }
            
            if metadata.expires_at:
                status["days_until_expiry"] = (metadata.expires_at - now).days
            
            return status
    
    def list_secrets(self) -> Dict[str, Dict]:
        """List all managed secrets with status."""
        return {key: self.get_secret_status(key) for key in self._metadata.keys()}
    
    def revoke_secret(self, key: str) -> bool:
        """Revoke and remove a secret."""
        with self._lock:
            if key in self._secrets:
                del self._secrets[key]
            if key in self._metadata:
                del self._metadata[key]
            if key in self._policies:
                del self._policies[key]
            
            self._save_state()
            logger.info(f"Revoked secret: {key}")
            return True


# Convenience function for getting secrets
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a secret value (convenience function)."""
    manager = SecretManager()
    return manager.get_secret(key, default)


def require_secret(key: str):
    """Decorator to inject secrets into functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = SecretManager()
            secret_value = manager.get_secret(key)
            if secret_value is None:
                raise ValueError(f"Required secret '{key}' not found")
            kwargs[key] = secret_value
            return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test the secret manager
    manager = SecretManager(auto_rotation=False)
    
    # Register a test secret
    manager.register_secret(
        "test_api_key",
        "test_value_12345",
        rotation_policy=SecretRotationPolicy(interval_days=30)
    )
    
    # Retrieve
    value = manager.get_secret("test_api_key")
    print(f"Retrieved: {value}")
    
    # Check status
    status = manager.get_secret_status("test_api_key")
    print(f"Status: {json.dumps(status, indent=2)}")

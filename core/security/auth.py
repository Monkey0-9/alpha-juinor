
import jwt
import datetime
import os
import logging
from typing import Optional

logger = logging.getLogger("AUTH")

class SecurityManager:
    """
    Handles Institutional Authentication (JWT) and Zero-Trust access.
    Bridges the 'Security Hardening' gap.
    """
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET", "institutional-zero-trust-key")

    def generate_service_token(self, service_name: str) -> str:
        """Issue an identity token for an internal microservice."""
        payload = {
            "service": service_name,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24),
            "iat": datetime.datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def validate_token(self, token: str) -> Optional[dict]:
        """Verify token integrity."""
        try:
            return jwt.decode(token, self.secret_key, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            logger.error("Token expired.")
        except jwt.InvalidTokenError:
            logger.error("Invalid token attempt.")
        return None

def get_security_manager() -> SecurityManager:
    return SecurityManager()

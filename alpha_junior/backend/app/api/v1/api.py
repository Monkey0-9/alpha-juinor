"""
API Router - combines all endpoint modules
"""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, funds, investments, kyc, users, admin, market

api_router = APIRouter()

# Include all endpoint modules
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(funds.router, prefix="/funds", tags=["Funds"])
api_router.include_router(investments.router, prefix="/investments", tags=["Investments"])
api_router.include_router(kyc.router, prefix="/kyc", tags=["KYC"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
api_router.include_router(admin.router, prefix="/admin", tags=["Admin"])
api_router.include_router(market.router, prefix="/market", tags=["Market Data"])

# WebSocket endpoint (to be implemented)
# api_router.include_router(websocket.router, prefix="/ws")

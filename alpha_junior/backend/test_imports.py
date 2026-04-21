#!/usr/bin/env python3
"""Test script to verify all imports work correctly"""

import sys

def test_imports():
    errors = []
    
    # Test core imports
    try:
        from app.core.config import settings
        print("✓ config imported")
    except Exception as e:
        errors.append(f"✗ config: {e}")
    
    try:
        from app.core.security import verify_password, create_access_token
        print("✓ security imported")
    except Exception as e:
        errors.append(f"✗ security: {e}")
    
    # Test models
    models_to_test = [
        'app.models.base',
        'app.models.user',
        'app.models.fund',
        'app.models.investment',
        'app.models.kyc',
        'app.models.notifications',
        'app.models.audit',
        'app.models.watchlist',
        'app.models.messages',
        'app.models.reports',
    ]
    
    for model in models_to_test:
        try:
            __import__(model)
            print(f"✓ {model.split('.')[-1]} imported")
        except Exception as e:
            errors.append(f"✗ {model}: {e}")
    
    # Test API endpoints
    endpoints_to_test = [
        'app.api.deps',
        'app.api.v1.api',
        'app.api.v1.endpoints.auth',
        'app.api.v1.endpoints.funds',
        'app.api.v1.endpoints.investments',
        'app.api.v1.endpoints.kyc',
        'app.api.v1.endpoints.users',
        'app.api.v1.endpoints.admin',
        'app.api.v1.endpoints.market',
    ]
    
    for endpoint in endpoints_to_test:
        try:
            __import__(endpoint)
            print(f"✓ {endpoint.split('.')[-1]} imported")
        except Exception as e:
            errors.append(f"✗ {endpoint}: {e}")
    
    # Test services
    try:
        from app.services.market_data import market_data_service
        print("✓ market_data imported")
    except Exception as e:
        errors.append(f"✗ market_data: {e}")
    
    # Test main
    try:
        from app.main import app
        print("✓ main app imported")
    except Exception as e:
        errors.append(f"✗ main: {e}")
    
    print("\n" + "="*50)
    if errors:
        print(f"ERRORS FOUND: {len(errors)}")
        for error in errors:
            print(f"  {error}")
        return 1
    else:
        print("ALL IMPORTS SUCCESSFUL!")
        return 0

if __name__ == "__main__":
    sys.exit(test_imports())

import httpx
import os
import time
import json

# Load env for API Key
from dotenv import load_dotenv
load_dotenv()

BACKEND_URL = os.getenv("NEXUS_BACKEND_URL", "http://localhost:8001")
API_KEY = os.getenv("NEXUS_API_KEY", "")

def test_backend():
    print(f"Testing Backend at {BACKEND_URL}...")
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    
    try:
        with httpx.Client(timeout=30) as client:
            # 1. Test Health
            print("\n[1/4] Checking API Health...")
            resp = client.get(f"{BACKEND_URL}/api/alpaca/health")
            print(f"Status: {resp.status_code}")
            print(f"Body: {resp.text[:100]}")
            
            # 2. Test Brain Snapshot (The core of the UI dashboard)
            print("\n[2/4] Fetching Market Intelligence (Brain)...")
            resp = client.get(f"{BACKEND_URL}/api/monitor/brain", headers=headers)
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                analysis = data.get("analysis", {})
                print(f"Regime: {analysis.get('regime')}")
                print(f"Strategy: {analysis.get('selected_strategy')}")
                print(f"Agreement: {analysis.get('strategy_agreement'):.2%}")
            else:
                print(f"FAIL: {resp.text}")

            # 3. Test Account
            print("\n[3/4] Checking Account Status...")
            resp = client.get(f"{BACKEND_URL}/api/alpaca/account", headers=headers)
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                acc = resp.json()
                print(f"Equity: ${acc.get('portfolio_value', 0)}")
                print(f"Mode: {'Simulated' if acc.get('simulated') else 'Live/Paper'}")

            # 4. Test Security (Unauthorized access to mutation)
            if API_KEY:
                print("\n[4/4] Verifying Security (Unauthorized POST)...")
                resp = client.post(f"{BACKEND_URL}/api/alpaca/order", json={})
                if resp.status_code == 401:
                    print("SUCCESS: Endpoint is secured (401 Unauthorized)")
                else:
                    print(f"WARNING: Endpoint returned {resp.status_code} instead of 401")
            else:
                print("\n[4/4] Skipping security test (No API Key configured)")

    except Exception as e:
        print(f"FATAL ERROR during test: {e}")

if __name__ == "__main__":
    test_backend()

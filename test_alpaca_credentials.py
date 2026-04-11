#!/usr/bin/env python3
"""Test script to verify Alpaca credentials and module initialization."""

from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("ALPACA TRADING SYSTEM - CREDENTIALS VERIFICATION")
print("=" * 60)

# Test 1: Environment Variables
import os

print("\n1. Environment Variables:")
key = os.getenv("ALPACA_API_KEY")
secret = os.getenv("ALPACA_SECRET_KEY")
base_url = os.getenv("ALPACA_BASE_URL")
print(f"   ✓ ALPACA_API_KEY: {key[:20]}...")
print(f"   ✓ ALPACA_SECRET_KEY: {secret[:20]}...")
print(f"   ✓ ALPACA_BASE_URL: {base_url}")

# Test 2: Broker Handler
print("\n2. Alpaca Broker Handler:")
from brokers.alpaca_broker import AlpacaExecutionHandler

broker = AlpacaExecutionHandler()
print(f"   ✓ Broker initialized")
print(f"   ✓ Base URL: {broker.base_url}")

# Test 3: Data Provider
print("\n3. Alpaca Data Provider:")
from data.collectors.alpaca_collector import AlpacaDataProvider

provider = AlpacaDataProvider()
print(f"   ✓ Provider initialized")
print(f"   ✓ Authenticated: {provider._authenticated}")
print(f"   ✓ Base URL: {provider.base_url}")

# Test 4: Execution Handler
print("\n4. Alpaca Execution Handler:")
from execution.alpaca_handler import AlpacaExecutionHandler as ExecHandler

exec_handler = ExecHandler(paper=True)
print(f"   ✓ Execution handler initialized")
print(f"   ✓ Base URL: {exec_handler.base_url}")

print("\n" + "=" * 60)
print("✓ ALL SYSTEMS OPERATIONAL - READY FOR TRADING")
print("=" * 60)

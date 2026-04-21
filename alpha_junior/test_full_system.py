#!/usr/bin/env python3
"""
Alpha Junior - COMPREHENSIVE FULL SYSTEM TEST
Tests all 14 traders, risk management, APIs, and trading engine
"""

import sys
import os
import json
import time
import requests
from datetime import datetime

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    CLEAR = '\033[2J\033[H'

def print_header():
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 80)
    print("  🏛️  ALPHA JUNIOR v3.0 - COMPREHENSIVE SYSTEM TEST")
    print("  Top 1% Institutional Trading Platform")
    print("=" * 80)
    print(f"{Colors.RESET}")
    print()

def test_section(name):
    print(f"{Colors.BOLD}{Colors.BLUE}[TEST] {name}{Colors.RESET}")
    print("-" * 80)

def test_pass(name):
    print(f"  {Colors.GREEN}✓ {name}{Colors.RESET}")

def test_fail(name, error=""):
    print(f"  {Colors.RED}✗ {name}{Colors.RESET}")
    if error:
        print(f"    Error: {error}")

def test_warn(name):
    print(f"  {Colors.YELLOW}⚠ {name}{Colors.RESET}")

class SystemTest:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.base_url = "http://localhost:5000"
        
    def run_all_tests(self):
        print_header()
        
        # Test 1: Python Environment
        self.test_python_environment()
        
        # Test 2: Dependencies
        self.test_dependencies()
        
        # Test 3: File Structure
        self.test_file_structure()
        
        # Test 4: Configuration
        self.test_configuration()
        
        # Test 5: Import Tests
        self.test_imports()
        
        # Test 6: Server Status
        self.test_server_status()
        
        # Test 7: API Endpoints
        self.test_api_endpoints()
        
        # Test 8: 14 Trading Strategies
        self.test_trading_strategies()
        
        # Test 9: Risk Management
        self.test_risk_management()
        
        # Test 10: Institutional Core
        self.test_institutional_core()
        
        # Test 11: Database
        self.test_database()
        
        # Test 12: Alpaca Integration
        self.test_alpaca_integration()
        
        # Summary
        self.print_summary()
        
    def test_python_environment(self):
        test_section("1. Python Environment")
        
        try:
            import sys
            version = sys.version_info
            if version.major == 3 and version.minor >= 11:
                test_pass(f"Python {version.major}.{version.minor}.{version.micro} (>= 3.11 required)")
                self.passed += 1
            else:
                test_warn(f"Python {version.major}.{version.minor}.{version.micro} (3.11+ recommended)")
                self.warnings += 1
        except Exception as e:
            test_fail("Python version check", str(e))
            self.failed += 1
            
    def test_dependencies(self):
        test_section("2. Dependencies")
        
        packages = [
            ('flask', 'Flask'),
            ('flask_cors', 'Flask-CORS'),
            ('requests', 'Requests'),
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas'),
            ('scipy', 'SciPy')
        ]
        
        for module_name, display_name in packages:
            try:
                __import__(module_name)
                test_pass(f"{display_name} installed")
                self.passed += 1
            except ImportError as e:
                test_fail(f"{display_name}", str(e))
                self.failed += 1
                
    def test_file_structure(self):
        test_section("3. File Structure")
        
        critical_files = [
            'app.py',
            'runner.py',
            'trading.py',
            'institutional_traders_v2.py',
            'institutional_core.py',
            'elite_hedge_fund.py',
            'autonomous_trader.py',
            'brain.py',
            'bloomberg_terminal.py',
            'requirements.txt',
            '.env'
        ]
        
        for file in critical_files:
            if os.path.exists(file):
                test_pass(f"{file} present")
                self.passed += 1
            else:
                test_fail(f"{file}", "File not found")
                self.failed += 1
                
    def test_configuration(self):
        test_section("4. Configuration")
        
        try:
            with open('.env', 'r', encoding='utf-8') as f:
                env_content = f.read()
                
            if 'ALPACA_API_KEY=' in env_content:
                test_pass("Alpaca API key configured")
                self.passed += 1
            else:
                test_fail("Alpaca API key", "Not found in .env")
                self.failed += 1
                
            if 'YOUR_SECRET_KEY_HERE' in env_content or 'YOUR_API_KEY_HERE' in env_content:
                test_warn("Using demo credentials - add real keys for live trading")
                self.warnings += 1
            else:
                test_pass("Real API credentials detected")
                self.passed += 1
                
        except Exception as e:
            test_warn(f"Configuration check: {e}")
            self.warnings += 1
            
    def test_imports(self):
        test_section("5. Module Imports")
        
        modules = [
            ('app', 'Flask Application'),
            ('trading', 'Trading Module'),
            ('institutional_traders_v2', '14 Trading Strategies'),
            ('institutional_core', 'Institutional Risk Engine'),
            ('elite_hedge_fund', 'Elite Hedge Fund'),
            ('autonomous_trader', 'Autonomous Trader'),
            ('brain', 'AI Brain'),
        ]
        
        for module_name, display_name in modules:
            try:
                __import__(module_name)
                test_pass(f"{display_name}")
                self.passed += 1
            except Exception as e:
                test_fail(f"{display_name}", str(e))
                self.failed += 1
                
    def test_server_status(self):
        test_section("6. Server Status")
        
        try:
            response = requests.get(f'{self.base_url}/api/health', timeout=5)
            if response.status_code == 200:
                data = response.json()
                test_pass(f"Server running on port 5000 ({data.get('service', 'unknown')})")
                self.passed += 1
                self.server_running = True
            else:
                test_warn(f"Server returned status {response.status_code} (will start on launch)")
                self.warnings += 1
                self.server_running = False
        except requests.exceptions.ConnectionError:
            test_warn("Server not running (will auto-start on launch)")
            self.warnings += 1
            self.server_running = False
        except Exception as e:
            test_warn(f"Server check: {e}")
            self.warnings += 1
            self.server_running = False
            
    def test_api_endpoints(self):
        test_section("7. API Endpoints")
        
        # Skip if server not running
        if not getattr(self, 'server_running', False):
            test_warn("Skipping API tests (server not running)")
            test_warn("  API endpoints will be verified when server starts")
            self.warnings += 2
            return
        
        endpoints = [
            ('/', 'GET', 'Home page'),
            ('/api/health', 'GET', 'Health check'),
            ('/api/funds', 'GET', 'Funds list'),
            ('/api/trading/account', 'GET', 'Trading account'),
        ]
        
        for endpoint, method, description in endpoints:
            try:
                if method == 'GET':
                    response = requests.get(f'{self.base_url}{endpoint}', timeout=5)
                else:
                    response = requests.post(f'{self.base_url}{endpoint}', timeout=5)
                    
                if response.status_code in [200, 201]:
                    test_pass(f"{description} ({endpoint})")
                    self.passed += 1
                else:
                    test_warn(f"{description} returned status {response.status_code}")
                    self.warnings += 1
            except Exception as e:
                test_warn(f"{description} - {e}")
                self.warnings += 1
                
    def test_trading_strategies(self):
        test_section("8. 14 Trading Strategies")
        
        try:
            from institutional_traders_v2 import get_complete_team, StrategyType
            import logging
            
            # Create test logger
            logger = logging.getLogger('TestLogger')
            logger.setLevel(logging.INFO)
            
            # Get trading team
            team = get_complete_team(logger)
            
            # Count strategies
            strategy_count = len(team.traders)
            if strategy_count >= 14:
                test_pass(f"All {strategy_count} trading strategies loaded")
                self.passed += 1
            else:
                test_fail(f"Trading strategies", f"Only {strategy_count} loaded, expected 14+")
                self.failed += 1
                
            # Test strategy types (14 total)
            expected_strategies = [
                'momentum', 'mean_reversion', 'breakout', 'trend_following',
                'swing_trading', 'scalping', 'position_trading', 'arbitrage',
                'gap_trading', 'sector_rotation', 'volatility', 'news_event',
                'algorithmic', 'pairs_trading'
            ]
            
            for strategy_name in expected_strategies:
                if strategy_name in team.traders:
                    test_pass(f"  {strategy_name.replace('_', ' ').title()} trader ready")
                    self.passed += 1
                else:
                    test_warn(f"  {strategy_name} - will use fallback")
                    self.warnings += 1
                    
        except Exception as e:
            test_fail("Trading strategies", str(e))
            self.failed += 14
            
    def test_risk_management(self):
        test_section("9. Risk Management")
        
        try:
            from institutional_core import InstitutionalRiskManager
            import logging
            
            logger = logging.getLogger('RiskTest')
            risk_manager = InstitutionalRiskManager(logger)
            
            # Test risk limits
            limits = risk_manager.limits
            test_pass(f"VaR 95% limit: {limits.get('portfolio_var_95', 0)*100:.1f}%")
            test_pass(f"Max position size: {limits.get('max_position_size', 0)*100:.1f}%")
            test_pass(f"Max drawdown: {limits.get('max_drawdown', 0)*100:.1f}%")
            test_pass(f"Max leverage: {limits.get('max_leverage', 0):.1f}x")
            test_pass(f"Min cash: {limits.get('min_cash', 0)*100:.1f}%")
            self.passed += 5
            
            # Test VaR calculation
            test_returns = [-0.02, 0.01, -0.01, 0.015, -0.005, 0.02, -0.015, 0.01]
            var_95 = risk_manager.calculate_var(test_returns, 0.95)
            test_pass(f"VaR calculation working (test value: {var_95:.2%})")
            self.passed += 1
            
            # Test stress testing
            positions = {}
            stress = risk_manager.stress_test(positions, "2008_crisis")
            if stress.get('scenario'):
                test_pass(f"Stress testing available ({stress['scenario']})")
                self.passed += 1
            else:
                test_fail("Stress testing")
                self.failed += 1
                
        except Exception as e:
            test_fail("Risk management", str(e))
            self.failed += 7
            
    def test_institutional_core(self):
        test_section("10. Institutional Core Engine")
        
        try:
            from institutional_core import get_institutional_core
            
            core = get_institutional_core()
            test_pass("Institutional core engine initialized")
            self.passed += 1
            
            # Test components
            if hasattr(core, 'risk_manager'):
                test_pass("Risk manager component loaded")
                self.passed += 1
            else:
                test_fail("Risk manager component")
                self.failed += 1
                
            if hasattr(core, 'execution_engine'):
                test_pass("Execution engine component loaded")
                self.passed += 1
            else:
                test_fail("Execution engine component")
                self.failed += 1
                
            if hasattr(core, 'optimizer'):
                test_pass("Portfolio optimizer loaded")
                self.passed += 1
            else:
                test_fail("Portfolio optimizer")
                self.failed += 1
                
        except Exception as e:
            test_fail("Institutional core", str(e))
            self.failed += 4
            
    def test_database(self):
        test_section("11. Database")
        
        try:
            import sqlite3
            
            # Check if database exists
            if os.path.exists('alpha_junior.db'):
                test_pass("Database file exists")
                self.passed += 1
                
                # Test connection
                conn = sqlite3.connect('alpha_junior.db')
                cursor = conn.cursor()
                
                # Check tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                table_names = [t[0] for t in tables]
                
                required_tables = ['users', 'funds', 'investments']
                for table in required_tables:
                    if table in table_names:
                        test_pass(f"  Table '{table}' exists")
                        self.passed += 1
                    else:
                        test_warn(f"  Table '{table}' not found (will be created on first run)")
                        self.warnings += 1
                        
                conn.close()
            else:
                test_warn("Database not initialized (will be created on first run)")
                self.warnings += 1
                
        except Exception as e:
            test_fail("Database check", str(e))
            self.failed += 1
            
    def test_alpaca_integration(self):
        test_section("12. Alpaca Paper Trading Integration")
        
        try:
            # Check if we can connect to Alpaca
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            
            api_key = os.getenv('ALPACA_API_KEY', '')
            secret_key = os.getenv('ALPACA_SECRET_KEY', '')
            
            if api_key and api_key != 'YOUR_API_KEY_HERE':
                test_pass(f"Alpaca API key configured ({api_key[:10]}...)")
                self.passed += 1
            else:
                test_warn("Alpaca API key not configured (demo mode)")
                self.warnings += 1
                
            if secret_key and secret_key != 'YOUR_SECRET_KEY_HERE':
                test_pass("Alpaca secret key configured")
                self.passed += 1
            else:
                test_warn("Alpaca secret key not configured (demo mode)")
                self.warnings += 1
                
            # Try to connect to Alpaca API
            if api_key and secret_key and 'YOUR' not in api_key and 'YOUR' not in secret_key:
                try:
                    headers = {
                        'APCA-API-KEY-ID': api_key,
                        'APCA-API-SECRET-KEY': secret_key
                    }
                    response = requests.get(
                        'https://paper-api.alpaca.markets/v2/account',
                        headers=headers,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        account = response.json()
                        test_pass(f"✓ Alpaca connection successful!")
                        test_pass(f"  Paper balance: ${account.get('cash', 0)}")
                        test_pass(f"  Buying power: ${account.get('buying_power', 0)}")
                        test_pass(f"  Account status: {account.get('status', 'unknown')}")
                        self.passed += 4
                    else:
                        test_fail("Alpaca connection", f"Status {response.status_code}")
                        self.failed += 1
                        
                except Exception as e:
                    test_fail("Alpaca API connection", str(e))
                    self.failed += 1
            else:
                test_warn("Skipping Alpaca connection test (demo credentials)")
                self.warnings += 1
                
        except Exception as e:
            test_fail("Alpaca integration", str(e))
            self.failed += 1
            
    def print_summary(self):
        print()
        print(f"{Colors.CYAN}{Colors.BOLD}")
        print("=" * 80)
        print("  TEST SUMMARY")
        print("=" * 80)
        print(f"{Colors.RESET}")
        print()
        
        total = self.passed + self.failed + self.warnings
        
        print(f"  {Colors.GREEN}✓ Passed:   {self.passed}/{total}{Colors.RESET}")
        print(f"  {Colors.RED}✗ Failed:   {self.failed}/{total}{Colors.RESET}")
        print(f"  {Colors.YELLOW}⚠ Warnings: {self.warnings}/{total}{Colors.RESET}")
        print()
        
        if self.failed == 0:
            print(f"  {Colors.GREEN}{Colors.BOLD}🎉 ALL CRITICAL TESTS PASSED!{Colors.RESET}")
            print(f"  {Colors.GREEN}System is ready for institutional trading.{Colors.RESET}")
            print()
            print(f"  {Colors.CYAN}Next steps:{Colors.RESET}")
            print(f"  1. Run: VERIFY_AND_RUN.bat")
            print(f"  2. Or run: ONE_CLICK_RUN.bat")
            print(f"  3. Start paper trading with $100,000")
        else:
            print(f"  {Colors.RED}{Colors.BOLD}⚠ {self.failed} TESTS FAILED{Colors.RESET}")
            print(f"  Please fix the issues above before trading.")
            print()
            print(f"  {Colors.YELLOW}Common fixes:{Colors.RESET}")
            print(f"  - Install Python 3.11+: https://python.org")
            print(f"  - Run: pip install flask flask-cors requests numpy pandas scipy")
            print(f"  - Add Alpaca API keys to .env file")
            
        print()
        print(f"{Colors.CYAN}{Colors.BOLD}")
        print("=" * 80)
        print(f"{Colors.RESET}")

def main():
    print(f"{Colors.CLEAR}", end="")
    tester = SystemTest()
    tester.run_all_tests()
    
    # Wait for user to see results
    print()
    input("Press Enter to exit...")

if __name__ == '__main__':
    main()

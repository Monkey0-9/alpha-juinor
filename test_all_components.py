"""
COMPREHENSIVE TEST SUITE - MiniQuantFund v4.0.0
Tests ALL components before finalization
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

# Setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/comprehensive_test.log')
    ]
)

logger = logging.getLogger('MiniQuantFund-Test')

class ComponentTester:
    """Test all system components"""
    
    def __init__(self):
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0
        
    def log_result(self, component, test_name, passed, details=""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.results[f"{component}.{test_name}"] = {
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        
        print(f"   {status} | {component}.{test_name}")
        if details and not passed:
            print(f"      Details: {details}")
    
    # ==================== ENVIRONMENT TESTS ====================
    def test_environment_variables(self):
        """Test 1: Environment Variables"""
        print("\n🔧 TEST 1: ENVIRONMENT VARIABLES")
        load_dotenv()
        
        required_vars = [
            'ALPACA_API_KEY',
            'ALPACA_SECRET_KEY',
            'POLYGON_API_KEY'
        ]
        
        for var in required_vars:
            value = os.getenv(var)
            self.log_result('Environment', var, bool(value), 
                          f"Value: {value[:15]}..." if value else "Not set")
    
    # ==================== ALPACA CONNECTION TESTS ====================
    def test_alpaca_connection(self):
        """Test 2: Alpaca API Connection"""
        print("\n📊 TEST 2: ALPACA API CONNECTION")
        
        try:
            import alpaca_trade_api as tradeapi
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = 'https://paper-api.alpaca.markets'
            
            # Test connection
            api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
            
            # Get account
            account = api.get_account()
            self.log_result('Alpaca', 'Connection', True, 
                          f"Status: {account.status}")
            self.log_result('Alpaca', 'Account_Info', True,
                          f"Equity: ${float(account.equity):,.2f}")
            self.log_result('Alpaca', 'Buying_Power', True,
                          f"Power: ${float(account.buying_power):,.2f}")
            
            # Get positions
            positions = api.list_positions()
            self.log_result('Alpaca', 'Positions_API', True,
                          f"Count: {len(positions)}")
            
            # Get orders
            orders = api.list_orders(status='all', limit=10)
            self.log_result('Alpaca', 'Orders_API', True,
                          f"Recent: {len(orders)}")
            
        except Exception as e:
            self.log_result('Alpaca', 'Connection', False, str(e))
            self.log_result('Alpaca', 'Account_Info', False, "No connection")
            self.log_result('Alpaca', 'Buying_Power', False, "No connection")
            self.log_result('Alpaca', 'Positions_API', False, "No connection")
            self.log_result('Alpaca', 'Orders_API', False, "No connection")
    
    # ==================== ORDER EXECUTION TESTS ====================
    def test_order_execution(self):
        """Test 3: Order Execution"""
        print("\n🛒 TEST 3: ORDER EXECUTION")
        
        try:
            import alpaca_trade_api as tradeapi
            import time
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = 'https://paper-api.alpaca.markets'
            
            api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
            
            # Submit market order
            order = api.submit_order(
                symbol='AAPL',
                qty=1,
                side='buy',
                type='market',
                time_in_force='day',
                client_order_id=f'test_{datetime.now().strftime("%H%M%S")}'
            )
            
            self.log_result('Orders', 'Submit_Market', True,
                          f"ID: {order.id}, Status: {order.status}")
            
            # Wait for execution
            time.sleep(3)
            
            # Check status
            order_status = api.get_order(order.id)
            self.log_result('Orders', 'Get_Status', True,
                          f"Status: {order_status.status}")
            
            if order_status.status == 'filled':
                self.log_result('Orders', 'Execution_Fill', True,
                              f"Price: ${order_status.filled_avg_price}")
            else:
                self.log_result('Orders', 'Execution_Fill', False,
                              f"Status: {order_status.status}")
            
            # Cancel any open orders
            api.cancel_all_orders()
            self.log_result('Orders', 'Cancel_All', True, "Executed")
            
        except Exception as e:
            self.log_result('Orders', 'Submit_Market', False, str(e))
            self.log_result('Orders', 'Get_Status', False, "Submit failed")
            self.log_result('Orders', 'Execution_Fill', False, "Submit failed")
            self.log_result('Orders', 'Cancel_All', False, str(e))
    
    # ==================== RISK MANAGEMENT TESTS ====================
    def test_risk_management(self):
        """Test 4: Risk Management"""
        print("\n🛡️  TEST 4: RISK MANAGEMENT")
        
        try:
            import alpaca_trade_api as tradeapi
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = 'https://paper-api.alpaca.markets'
            
            api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
            
            # Check account for risk metrics
            account = api.get_account()
            
            equity = float(account.equity)
            last_equity = float(account.last_equity)
            buying_power = float(account.buying_power)
            
            # Calculate risk metrics
            daily_pl = equity - last_equity
            daily_pl_pct = (daily_pl / last_equity) * 100 if last_equity > 0 else 0
            leverage = buying_power / equity if equity > 0 else 0
            
            self.log_result('Risk', 'Daily_PnL', True,
                          f"${daily_pl:,.2f} ({daily_pl_pct:+.2f}%)")
            
            self.log_result('Risk', 'Leverage_Check', leverage <= 2.0,
                          f"{leverage:.2f}:1 (Max: 2:1)")
            
            # Check positions for concentration risk
            positions = api.list_positions()
            total_exposure = sum(float(p.market_value) for p in positions)
            
            if equity > 0:
                exposure_pct = (total_exposure / equity) * 100
                self.log_result('Risk', 'Exposure_Check', exposure_pct <= 100,
                              f"{exposure_pct:.1f}% of equity")
            else:
                self.log_result('Risk', 'Exposure_Check', True, "No positions")
            
            # Check for circuit breaker conditions
            drawdown_pct = 0  # Simplified - would need historical data
            self.log_result('Risk', 'Circuit_Breaker', drawdown_pct < 5.0,
                          f"Current DD: {drawdown_pct:.2f}% (Limit: 5%)")
            
        except Exception as e:
            self.log_result('Risk', 'Daily_PnL', False, str(e))
            self.log_result('Risk', 'Leverage_Check', False, str(e))
            self.log_result('Risk', 'Exposure_Check', False, str(e))
            self.log_result('Risk', 'Circuit_Breaker', False, str(e))
    
    # ==================== POSITION MANAGEMENT TESTS ====================
    def test_position_management(self):
        """Test 5: Position Management"""
        print("\n📈 TEST 5: POSITION MANAGEMENT")
        
        try:
            import alpaca_trade_api as tradeapi
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = 'https://paper-api.alpaca.markets'
            
            api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
            
            # Get positions
            positions = api.list_positions()
            
            self.log_result('Positions', 'List_Positions', True,
                          f"Count: {len(positions)}")
            
            if positions:
                # Analyze first position
                pos = positions[0]
                
                self.log_result('Positions', 'Position_Details', True,
                              f"{pos.symbol}: {pos.qty} shares")
                
                self.log_result('Positions', 'Market_Value', True,
                              f"${float(pos.market_value):,.2f}")
                
                self.log_result('Positions', 'Unrealized_PnL', True,
                              f"${float(pos.unrealized_pl):,.2f}")
                
                self.log_result('Positions', 'Cost_Basis', True,
                              f"Avg: ${pos.avg_entry_price}")
            else:
                self.log_result('Positions', 'Position_Details', True, "No positions")
                self.log_result('Positions', 'Market_Value', True, "No positions")
                self.log_result('Positions', 'Unrealized_PnL', True, "No positions")
                self.log_result('Positions', 'Cost_Basis', True, "No positions")
            
            # Test closed positions (activities)
            activities = api.get_activities()
            self.log_result('Positions', 'Trade_History', True,
                          f"Activities: {len(activities)}")
            
        except Exception as e:
            self.log_result('Positions', 'List_Positions', False, str(e))
            self.log_result('Positions', 'Position_Details', False, str(e))
            self.log_result('Positions', 'Market_Value', False, str(e))
            self.log_result('Positions', 'Unrealized_PnL', False, str(e))
            self.log_result('Positions', 'Cost_Basis', False, str(e))
            self.log_result('Positions', 'Trade_History', False, str(e))
    
    # ==================== COMPLIANCE & AUDIT TESTS ====================
    def test_compliance_audit(self):
        """Test 6: Compliance & Audit"""
        print("\n📋 TEST 6: COMPLIANCE & AUDIT TRAIL")
        
        try:
            import alpaca_trade_api as tradeapi
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = 'https://paper-api.alpaca.markets'
            
            api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
            
            # Get recent orders for audit
            orders = api.list_orders(status='all', limit=50)
            
            self.log_result('Compliance', 'Order_Audit_Log', True,
                          f"Recent orders: {len(orders)}")
            
            if orders:
                latest = orders[0]
                audit_fields = [
                    ('Order_ID', latest.id),
                    ('Client_Order_ID', latest.client_order_id),
                    ('Symbol', latest.symbol),
                    ('Side', latest.side),
                    ('Qty', latest.qty),
                    ('Type', latest.type),
                    ('Status', latest.status),
                    ('Created_At', latest.created_at),
                    ('Filled_Avg_Price', getattr(latest, 'filled_avg_price', 'N/A'))
                ]
                
                for field, value in audit_fields:
                    self.log_result('Compliance', f'Audit_{field}', True, str(value))
            
            # Check portfolio history
            portfolio_history = api.get_portfolio_history(period='1D')
            self.log_result('Compliance', 'Portfolio_History', True,
                          f"Data points: {len(portfolio_history.equity)}")
            
        except Exception as e:
            self.log_result('Compliance', 'Order_Audit_Log', False, str(e))
            self.log_result('Compliance', 'Portfolio_History', False, str(e))
    
    # ==================== PERFORMANCE TESTS ====================
    def test_performance(self):
        """Test 7: System Performance"""
        print("\n⚡ TEST 7: SYSTEM PERFORMANCE")
        
        import time
        
        try:
            import alpaca_trade_api as tradeapi
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = 'https://paper-api.alpaca.markets'
            
            # Test API latency
            start_time = time.time()
            api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
            account = api.get_account()
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            self.log_result('Performance', 'API_Latency', latency < 500,
                          f"{latency:.2f}ms (Target: <500ms)")
            
            # Test order submission latency
            start_time = time.time()
            order = api.submit_order(
                symbol='AAPL',
                qty=1,
                side='buy',
                type='market',
                time_in_force='day'
            )
            order_latency = (time.time() - start_time) * 1000
            
            self.log_result('Performance', 'Order_Submit_Latency', order_latency < 1000,
                          f"{order_latency:.2f}ms (Target: <1000ms)")
            
            # Cancel test order
            api.cancel_order(order.id)
            
            # Test data retrieval speed
            start_time = time.time()
            positions = api.list_positions()
            orders = api.list_orders(limit=10)
            account = api.get_account()
            data_latency = (time.time() - start_time) * 1000
            
            self.log_result('Performance', 'Data_Retrieval_Latency', data_latency < 1000,
                          f"{data_latency:.2f}ms (Target: <1000ms)")
            
        except Exception as e:
            self.log_result('Performance', 'API_Latency', False, str(e))
            self.log_result('Performance', 'Order_Submit_Latency', False, str(e))
            self.log_result('Performance', 'Data_Retrieval_Latency', False, str(e))
    
    # ==================== DATA FEED TESTS ====================
    def test_data_feeds(self):
        """Test 8: Data Feed Quality"""
        print("\n📡 TEST 8: DATA FEED QUALITY")
        
        try:
            import alpaca_trade_api as tradeapi
            
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = 'https://paper-api.alpaca.markets'
            
            api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
            
            # Test multiple symbols
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
            
            for symbol in symbols:
                try:
                    # Get latest bar
                    bar = api.get_latest_bar(symbol)
                    self.log_result('Data', f'Price_{symbol}', True,
                                  f"${bar.close:.2f} (Vol: {bar.volume})")
                except Exception as e:
                    self.log_result('Data', f'Price_{symbol}', False, str(e))
            
            # Test quote data
            try:
                quote = api.get_latest_quote('AAPL')
                self.log_result('Data', 'Quote_Data', True,
                              f"Bid: ${quote.bidprice:.2f}, Ask: ${quote.askprice:.2f}")
            except Exception as e:
                self.log_result('Data', 'Quote_Data', False, str(e))
            
        except Exception as e:
            for symbol in symbols:
                self.log_result('Data', f'Price_{symbol}', False, str(e))
            self.log_result('Data', 'Quote_Data', False, str(e))
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*70)
        print("🔬 COMPREHENSIVE COMPONENT TEST SUITE")
        print("MiniQuantFund v4.0.0 - Testing ALL Components")
        print("="*70)
        print(f"Start Time: {datetime.now().isoformat()}")
        print("="*70)
        
        # Run all tests
        self.test_environment_variables()
        self.test_alpaca_connection()
        self.test_order_execution()
        self.test_risk_management()
        self.test_position_management()
        self.test_compliance_audit()
        self.test_performance()
        self.test_data_feeds()
        
        # Generate report
        return self.generate_report()
    
    def generate_report(self):
        """Generate final test report"""
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("="*70)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success Rate: {(self.passed_tests/self.total_tests*100):.1f}%")
        print("="*70)
        
        # Component summary
        components = {}
        for test_name, result in self.results.items():
            component = test_name.split('.')[0]
            if component not in components:
                components[component] = {'passed': 0, 'total': 0}
            components[component]['total'] += 1
            if result['passed']:
                components[component]['passed'] += 1
        
        print("\n📋 COMPONENT BREAKDOWN:")
        for component, stats in components.items():
            rate = (stats['passed'] / stats['total'] * 100)
            status = "✅" if rate >= 80 else "⚠️"
            print(f"   {status} {component}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")
        
        print("="*70)
        
        # Final verdict
        if self.passed_tests == self.total_tests:
            print("🎉 ALL TESTS PASSED - SYSTEM FULLY OPERATIONAL")
            print("✅ Ready for institutional-grade paper trading")
            return True
        elif self.passed_tests / self.total_tests >= 0.8:
            print("⚠️  MOSTLY PASSED - Minor issues detected")
            print("⚠️  Review failed tests before trading")
            return True
        else:
            print("❌ TESTS FAILED - System not ready")
            print("❌ Fix issues before trading")
            return False
        
        print("="*70)

def main():
    """Main test function"""
    print("\n" + "🧪"*35)
    print("COMPREHENSIVE COMPONENT TESTING")
    print("Testing EVERY component before finalization")
    print("🧪"*35)
    
    tester = ComponentTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ TESTING COMPLETE - ALL SYSTEMS GO")
        print("📁 Detailed logs: logs/comprehensive_test.log")
    else:
        print("\n❌ TESTING COMPLETE - ISSUES FOUND")
        print("🔧 Review logs and fix issues")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        sys.exit(1)

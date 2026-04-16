"""
Complete Free System Test - Using only open-source and free resources
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreeSystemTester:
    """Test suite for free/open-source system"""

    def __init__(self):
        """Initialize tester"""
        self.results = {}
        self.start_time = time.time()

    def test_fpga_components(self):
        """Test FPGA VHDL components"""
        try:
            fpga_files = [
                'fpga/rtl/order_book.vhd',
                'fpga/rtl/matching_engine.vhd',
                'fpga/rtl/pcie_dma.vhd',
                'fpga/rtl/eth_mac.vhd',
                'fpga/rtl/top_level.vhd'
            ]

            missing_files = []
            for file_path in fpga_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)

            if missing_files:
                return {
                    'status': 'FAILED',
                    'error': f'Missing FPGA files: {missing_files}'
                }

            return {
                'status': 'PASSED',
                'message': 'All FPGA components present',
                'files': fpga_files
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_python_greeks(self):
        """Test pure Python Greeks calculator"""
        try:
            from mini_quant_fund.options.python_greeks import PurePythonGreeksCalculator

            calculator = PurePythonGreeksCalculator()

            # Test single calculation
            greeks = calculator.calculate_greeks(
                S=100.0, K=95.0, T=0.25, r=0.05, sigma=0.2, is_call=True
            )

            # Verify Greeks are reasonable
            if not (0 <= greeks.delta <= 1):
                return {
                    'status': 'FAILED',
                    'error': f'Invalid delta: {greeks.delta}'
                }

            if greeks.gamma < 0:
                return {
                    'status': 'FAILED',
                    'error': f'Invalid gamma: {greeks.gamma}'
                }

            # Test batch calculation
            batch_data = pd.DataFrame({
                'S': [100, 105, 110],
                'K': [95, 100, 105],
                'T': [0.25, 0.5, 0.75],
                'r': [0.05, 0.05, 0.05],
                'sigma': [0.2, 0.25, 0.3],
                'is_call': [True, False, True],
                'quantity': [100, 200, 150]
            })

            batch_results = calculator.calculate_batch_greeks(batch_data)

            if len(batch_results) != 3:
                return {
                    'status': 'FAILED',
                    'error': f'Batch calculation failed: expected 3, got {len(batch_results)}'
                }

            # Test portfolio calculation
            portfolio = [
                {'S': 100, 'K': 95, 'T': 0.25, 'r': 0.05, 'sigma': 0.2, 'is_call': True, 'quantity': 100},
                {'S': 100, 'K': 105, 'T': 0.25, 'r': 0.05, 'sigma': 0.2, 'is_call': False, 'quantity': 50}
            ]

            portfolio_greeks = calculator.calculate_portfolio_greeks(portfolio)

            return {
                'status': 'PASSED',
                'single_greeks': {
                    'delta': greeks.delta,
                    'gamma': greeks.gamma,
                    'theta': greeks.theta,
                    'vega': greeks.vega,
                    'rho': greeks.rho
                },
                'batch_size': len(batch_results),
                'portfolio_delta': portfolio_greeks['delta']
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_free_satellite_data(self):
        """Test free satellite data integration"""
        try:
            from mini_quant_fund.alternative_data.satellite.free_satellite import analyze_free_satellite_data

            # Test satellite data analysis
            location = {
                'lat': 40.7128,
                'lon': -74.0060,
                'name': 'Walmart NYC',
                'city': 'New York, NY'
            }

            result = analyze_free_satellite_data('WMT', location)

            if 'error' in result:
                return {
                    'status': 'FAILED',
                    'error': result['error']
                }

            # Verify expected fields
            expected_fields = ['ticker', 'location', 'data_sources', 'avg_yoy_growth', 'status']
            for field in expected_fields:
                if field not in result:
                    return {
                        'status': 'FAILED',
                        'error': f'Missing field: {field}'
                    }

            if result['status'] != 'FREE_DATA_SUCCESS':
                return {
                    'status': 'FAILED',
                    'error': f'Unexpected status: {result["status"]}'
                }

            return {
                'status': 'PASSED',
                'data_sources': result['data_sources'],
                'avg_yoy_growth': result['avg_yoy_growth'],
                'num_sources': len(result['data_sources'])
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_free_alternative_data(self):
        """Test free alternative data sources"""
        try:
            from mini_quant_fund.alternative_data.credit_card.free_spending import analyze_free_alternative_data

            # Test alternative data analysis
            result = analyze_free_alternative_data('WMT')

            if 'error' in result:
                return {
                    'status': 'FAILED',
                    'error': result['error']
                }

            # Verify expected fields
            expected_fields = ['ticker', 'data_sources', 'composite_sentiment_score', 'status']
            for field in expected_fields:
                if field not in result:
                    return {
                        'status': 'FAILED',
                        'error': f'Missing field: {field}'
                    }

            if result['status'] != 'FREE_DATA_SUCCESS':
                return {
                    'status': 'FAILED',
                    'error': f'Unexpected status: {result["status"]}'
                }

            return {
                'status': 'PASSED',
                'data_sources': result['data_sources'],
                'composite_sentiment_score': result['composite_sentiment_score'],
                'composite_growth_rate': result['composite_growth_rate']
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_free_market_data(self):
        """Test free market data sources"""
        try:
            from mini_quant_fund.market_data.free_market_data import get_free_market_data

            # Test market data
            result = get_free_market_data('AAPL', 'quote')

            if 'error' in result:
                return {
                    'status': 'FAILED',
                    'error': result['error']
                }

            # Verify expected fields
            expected_fields = ['symbol', 'price', 'volume', 'sources']
            for field in expected_fields:
                if field not in result:
                    return {
                        'status': 'FAILED',
                        'error': f'Missing field: {field}'
                    }

            if result['price'] <= 0:
                return {
                    'status': 'FAILED',
                    'error': f'Invalid price: {result["price"]}'
                }

            # Test historical data
            hist_result = get_free_market_data('AAPL', 'historical')

            if 'error' not in hist_result:
                return {
                    'status': 'PASSED',
                    'quote_data': {
                        'symbol': result['symbol'],
                        'price': result['price'],
                        'volume': result['volume'],
                        'sources': result['sources']
                    },
                    'historical_data_available': True
                }
            else:
                return {
                    'status': 'PASSED',
                    'quote_data': {
                        'symbol': result['symbol'],
                        'price': result['price'],
                        'volume': result['volume'],
                        'sources': result['sources']
                    },
                    'historical_data_available': False
                }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_alpha_factory(self):
        """Test alpha factory with free data"""
        try:
            from mini_quant_fund.alpha_platform.alpha_dsl import AlphaDSL

            # Create test data with required columns
            data = pd.DataFrame({
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 100),
                'open': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 102,
                'low': np.random.randn(100).cumsum() + 98
            })

            dsl = AlphaDSL(data)

            # Test alpha expressions
            expressions = [
                "(close - ts_mean(close, 20)) / ts_std(close, 20)",
                "rank(close - ts_mean(close, 10))",
                "ts_delta(close, 1) / ts_std(close, 20)"
            ]

            results = {}
            for expr in expressions:
                try:
                    signal = dsl.evaluate(expr)
                    results[expr] = {
                        'success': True,
                        'signal_length': len(signal),
                        'signal_mean': float(signal.mean()),
                        'signal_std': float(signal.std())
                    }
                except Exception as e:
                    results[expr] = {
                        'success': False,
                        'error': str(e)
                    }

            failed_expressions = [expr for expr, result in results.items() if not result['success']]

            if failed_expressions:
                return {
                    'status': 'FAILED',
                    'error': f'Failed expressions: {failed_expressions}',
                    'results': results
                }

            return {
                'status': 'PASSED',
                'expressions_tested': len(expressions),
                'results': results
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_execution_algorithms(self):
        """Test execution algorithms"""
        try:
            from mini_quant_fund.execution.algorithms.vwap import VWAPAlgorithm
            from mini_quant_fund.execution.algorithms.twap import TWAPAlgorithm

            # Test VWAP
            vwap = VWAPAlgorithm()
            vwap_slices = vwap.execute("AAPL", 10000, "buy", 8)

            if not vwap_slices:
                return {
                    'status': 'FAILED',
                    'error': 'VWAP returned no slices'
                }

            # Validate VWAP slices
            total_quantity = sum(slice.quantity for slice in vwap_slices)
            if abs(total_quantity - 10000) > 1:
                return {
                    'status': 'FAILED',
                    'error': f'VWAP quantity mismatch: {total_quantity} vs 10000'
                }

            # Test TWAP
            twap = TWAPAlgorithm()
            twap_slices = twap.execute("AAPL", 5000, "sell", 240)

            if not twap_slices:
                return {
                    'status': 'FAILED',
                    'error': 'TWAP returned no slices'
                }

            # Validate TWAP slices
            total_quantity = sum(slice.quantity for slice in twap_slices)
            if abs(total_quantity - 5000) > 1:
                return {
                    'status': 'FAILED',
                    'error': f'TWAP quantity mismatch: {total_quantity} vs 5000'
                }

            return {
                'status': 'PASSED',
                'vwap_slices': len(vwap_slices),
                'twap_slices': len(twap_slices)
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_etf_arbitrage(self):
        """Test ETF arbitrage engine"""
        try:
            from mini_quant_fund.etf_arbitrage.etf_engine import ETFArbitrageEngine

            engine = ETFArbitrageEngine()

            # Test arbitrage detection
            arb_opportunity = engine.detect_arbitrage(
                etf_price=100.50,
                nav=100.30,
                tca_cost=0.0002
            )

            if not arb_opportunity:
                return {
                    'status': 'FAILED',
                    'error': 'No arbitrage opportunity detected'
                }

            # Verify arbitrage fields
            expected_fields = ['action', 'expected_profit', 'premium_discount']
            for field in expected_fields:
                if not hasattr(arb_opportunity, field):
                    return {
                        'status': 'FAILED',
                        'error': f'Missing arbitrage field: {field}'
                    }

            return {
                'status': 'PASSED',
                'arbitrage_action': getattr(arb_opportunity, 'action', 'unknown'),
                'expected_profit': getattr(arb_opportunity, 'expected_profit', 0),
                'premium_discount': getattr(arb_opportunity, 'premium_discount', 0)
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_risk_management(self):
        """Test risk management system"""
        try:
            from mini_quant_fund.live_trading.zero_loss_guard import ZeroLossRiskController

            guard = ZeroLossRiskController()

            # Test execution validation
            test_cases = [
                {'expected_price': 100.00, 'actual_price': 100.00, 'side': 'buy'},  # Perfect
                {'expected_price': 100.00, 'actual_price': 100.001, 'side': 'buy'},  # 1 bps
                {'expected_price': 100.00, 'actual_price': 100.01, 'side': 'buy'},   # 10 bps
                {'expected_price': 100.00, 'actual_price': 99.99, 'side': 'sell'},   # 10 bps
            ]

            results = []
            for case in test_cases:
                valid = guard.validate_execution(
                    expected_price=case['expected_price'],
                    actual_price=case['actual_price'],
                    side=case['side']
                )
                results.append({
                    'case': case,
                    'valid': valid
                })

            # Should reject cases with >1bps slippage
            rejected_count = sum(1 for r in results if not r['valid'])

            if rejected_count < 2:
                return {
                    'status': 'FAILED',
                    'error': f"Risk validation failed: Expected at least 2 rejections, got {rejected_count}"
                }

            return {
                'status': 'PASSED',
                'test_cases': len(test_cases),
                'rejected_count': rejected_count,
                'validation_rate': (len(test_cases) - rejected_count) / len(test_cases)
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_free_broker_integration(self):
        """Test free broker integration"""
        try:
            from mini_quant_fund.brokers.free_broker import get_free_broker_integration

            # Test paper trading broker
            broker_result = get_free_broker_integration('paper')

            if 'error' in broker_result:
                return {
                    'status': 'FAILED',
                    'error': broker_result['error']
                }

            broker = broker_result['broker']

            # Test account information
            account = broker.get_account()

            if 'error' in account:
                return {
                    'status': 'FAILED',
                    'error': 'Failed to get account information'
                }

            # Test order placement
            order = broker.place_order('AAPL', 100, 'buy')

            if 'error' in order:
                return {
                    'status': 'FAILED',
                    'error': 'Failed to place order'
                }

            # Test portfolio summary
            portfolio = broker.get_portfolio_summary()

            if 'error' in portfolio:
                return {
                    'status': 'FAILED',
                    'error': 'Failed to get portfolio summary'
                }

            return {
                'status': 'PASSED',
                'broker_type': 'paper_trading',
                'initial_capital': portfolio.get('initial_capital'),
                'total_value': portfolio.get('total_value'),
                'total_return_pct': portfolio.get('total_return_pct')
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        try:
            from mini_quant_fund.options.python_greeks import benchmark_greeks_calculator

            # Run benchmark
            benchmark_results = benchmark_greeks_calculator()

            # Check if performance is reasonable
            if benchmark_results['single_calc_avg_ms'] > 1.0:  # 1ms per calculation
                return {
                    'status': 'FAILED',
                    'error': f'Single calculation too slow: {benchmark_results["single_calc_avg_ms"]:.3f}ms'
                }

            if benchmark_results['batch_1000_calc_ms'] > 100.0:  # 100ms for 1000 calculations
                return {
                    'status': 'FAILED',
                    'error': f'Batch calculation too slow: {benchmark_results["batch_1000_calc_ms"]:.3f}ms'
                }

            return {
                'status': 'PASSED',
                'single_calc_avg_ms': benchmark_results['single_calc_avg_ms'],
                'batch_1000_calc_ms': benchmark_results['batch_1000_calc_ms'],
                'performance_ratio': benchmark_results['performance_ratio']
            }

        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}

    def run_all_tests(self):
        """Run all free system tests"""
        tests = [
            ('FPGA Components', self.test_fpga_components),
            ('Python Greeks Calculator', self.test_python_greeks),
            ('Free Satellite Data', self.test_free_satellite_data),
            ('Free Alternative Data', self.test_free_alternative_data),
            ('Free Market Data', self.test_free_market_data),
            ('Alpha Factory', self.test_alpha_factory),
            ('Execution Algorithms', self.test_execution_algorithms),
            ('ETF Arbitrage', self.test_etf_arbitrage),
            ('Risk Management', self.test_risk_management),
            ('Free Broker Integration', self.test_free_broker_integration),
            ('Performance Benchmarks', self.test_performance_benchmarks)
        ]

        logger.info("Starting MiniQuantFund v3.0.0 Free System Test")
        logger.info("=" * 60)

        for test_name, test_func in tests:
            logger.info(f"Running {test_name}...")
            start_time = time.time()

            result = test_func()
            elapsed_time = time.time() - start_time

            self.results[test_name] = {
                **result,
                'elapsed_time': elapsed_time
            }

            if result['status'] == 'PASSED':
                logger.info(f"  {test_name}: PASSED")
            else:
                logger.error(f"  {test_name}: FAILED - {result.get('error', 'Unknown error')}")

        # Generate summary
        total_tests = len(tests)
        passed_tests = sum(1 for r in self.results.values() if r['status'] == 'PASSED')
        failed_tests = total_tests - passed_tests

        logger.info("=" * 60)
        logger.info("MINIQUANTFUND v3.0.0 FREE SYSTEM TEST COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.2f}%")
        logger.info(f"Elapsed Time: {time.time() - self.start_time:.2f}s")
        logger.info("=" * 60)

        if failed_tests > 0:
            logger.error("FAILED TESTS:")
            for test_name, result in self.results.items():
                if result['status'] == 'FAILED':
                    logger.error(f"  {test_name}: {result.get('error', 'Unknown error')}")

        # Generate report
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests/total_tests)*100,
                'elapsed_time': time.time() - self.start_time,
                'timestamp': datetime.now().isoformat()
            },
            'test_results': self.results,
            'system_status': 'OPERATIONAL' if failed_tests == 0 else 'NEEDS_FIXES'
        }

        # Save report
        with open('free_system_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        return report

if __name__ == "__main__":
    tester = FreeSystemTester()
    report = tester.run_all_tests()

    print(f"\nSystem Status: {report['system_status']}")
    print(f"Success Rate: {report['test_summary']['success_rate']:.2f}%")
    print(f"Report saved to: free_system_test_report.json")

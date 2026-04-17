#!/usr/bin/env python3
"""
SCALING AND LOAD TESTING FRAMEWORK
=================================

Implement comprehensive load testing and scaling validation.
This provides actual performance testing beyond theoretical benchmarks.

Features:
- Load testing for trading systems
- Scalability validation
- Performance benchmarking
- Stress testing
- Capacity planning
- Performance monitoring
"""

import asyncio
import time
import json
import os
import threading
import multiprocessing
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import requests
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Load test configuration"""
    test_name: str
    test_type: str  # load, stress, spike, volume, endurance

    # Test parameters
    concurrent_users: int
    test_duration: int  # seconds
    ramp_up_time: int  # seconds
    think_time: float  # seconds between requests

    # Target configuration
    target_url: str
    target_endpoints: List[str]
    request_data: Dict[str, Any]

    # Performance targets
    target_rps: int  # requests per second
    target_latency_ms: float
    target_error_rate: float  # percentage

    # Monitoring
    monitor_resources: bool = True
    monitor_network: bool = True
    monitor_database: bool = True


@dataclass
class LoadTestResult:
    """Load test result"""
    test_name: str
    test_type: str
    start_time: datetime
    end_time: datetime
    duration: float

    # Performance metrics
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_rate: float

    # Resource metrics
    cpu_usage: List[float]
    memory_usage: List[float]
    network_io: List[Dict[str, float]]

    # Analysis
    performance_score: float = 0.0
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ScalingMetric:
    """Scaling metric"""
    metric_name: str
    current_value: float
    target_value: float
    unit: str

    # Scaling analysis
    scaling_factor: float = 1.0
    is_linear: bool = True
    bottleneck_threshold: float = 0.8


class LoadTestingFramework:
    """
    Comprehensive load testing and scaling validation framework.

    This provides actual performance testing capabilities to validate
    the system's ability to handle institutional trading workloads.
    """

    def __init__(self):
        # Test configurations
        self.test_configs: Dict[str, LoadTestConfig] = {}

        # Test results
        self.test_results: List[LoadTestResult] = []

        # Scaling metrics
        self.scaling_metrics: Dict[str, ScalingMetric] = {}

        # Monitoring data
        self.monitoring_data = {
            'cpu': [],
            'memory': [],
            'network': [],
            'response_times': [],
            'error_rates': []
        }

        # Test execution
        self.is_running = False
        self.current_test = None

        # Initialize test configurations
        self._initialize_test_configs()

        logger.info("Load Testing Framework initialized")

    def _initialize_test_configs(self):
        """Initialize standard load test configurations"""

        # Trading Engine Load Test
        self.test_configs['trading_engine_load'] = LoadTestConfig(
            test_name='Trading Engine Load Test',
            test_type='load',
            concurrent_users=100,
            test_duration=300,  # 5 minutes
            ramp_up_time=60,    # 1 minute ramp-up
            think_time=0.1,
            target_url='http://localhost:8000',
            target_endpoints=[
                '/api/trading/signals',
                '/api/trading/execute',
                '/api/trading/status',
                '/api/trading/portfolio'
            ],
            request_data={
                'symbol': 'AAPL',
                'quantity': 100,
                'order_type': 'market'
            },
            target_rps=1000,
            target_latency_ms=50,
            target_error_rate=0.01
        )

        # Risk Manager Stress Test
        self.test_configs['risk_manager_stress'] = LoadTestConfig(
            test_name='Risk Manager Stress Test',
            test_type='stress',
            concurrent_users=500,
            test_duration=600,  # 10 minutes
            ramp_up_time=120,   # 2 minutes ramp-up
            think_time=0.05,
            target_url='http://localhost:8001',
            target_endpoints=[
                '/api/risk/position',
                '/api/risk/var',
                '/api/risk/portfolio',
                '/api/risk/alerts'
            ],
            request_data={
                'portfolio_value': 1000000,
                'risk_limit': 0.02
            },
            target_rps=2000,
            target_latency_ms=100,
            target_error_rate=0.05
        )

        # Data Processor Volume Test
        self.test_configs['data_processor_volume'] = LoadTestConfig(
            test_name='Data Processor Volume Test',
            test_type='volume',
            concurrent_users=200,
            test_duration=1800,  # 30 minutes
            ramp_up_time=300,    # 5 minutes ramp-up
            think_time=0.2,
            target_url='http://localhost:8002',
            target_endpoints=[
                '/api/data/process',
                '/api/data/validate',
                '/api/data/store',
                '/api/data/retrieve'
            ],
            request_data={
                'data_type': 'market_data',
                'symbols': ['AAPL', 'MSFT', 'GOOGL'],
                'timeframe': '1m'
            },
            target_rps=500,
            target_latency_ms=200,
            target_error_rate=0.02
        )

        # Alternative Data Spike Test
        self.test_configs['alternative_data_spike'] = LoadTestConfig(
            test_name='Alternative Data Spike Test',
            test_type='spike',
            concurrent_users=1000,
            test_duration=120,   # 2 minutes
            ramp_up_time=10,    # 10 seconds ramp-up
            think_time=0.01,
            target_url='http://localhost:8003',
            target_endpoints=[
                '/api/alternative/satellite',
                '/api/alternative/credit_card',
                '/api/alternative/app_store',
                '/api/alternative/supply_chain'
            ],
            request_data={
                'data_source': 'satellite',
                'region': 'us',
                'time_period': 'daily'
            },
            target_rps=5000,
            target_latency_ms=500,
            target_error_rate=0.1
        )

        # Endurance Test
        self.test_configs['endurance_test'] = LoadTestConfig(
            test_name='Endurance Test',
            test_type='endurance',
            concurrent_users=50,
            test_duration=7200,  # 2 hours
            ramp_up_time=300,    # 5 minutes ramp-up
            think_time=1.0,
            target_url='http://localhost:8000',
            target_endpoints=[
                '/api/trading/status',
                '/api/trading/portfolio',
                '/api/trading/performance'
            ],
            request_data={
                'check_interval': 60
            },
            target_rps=50,
            target_latency_ms=100,
            target_error_rate=0.001
        )

        logger.info(f"Initialized {len(self.test_configs)} test configurations")

    async def run_load_test(self, config_name: str) -> LoadTestResult:
        """Run load test with specified configuration"""
        try:
            config = self.test_configs.get(config_name)
            if not config:
                raise ValueError(f"Test configuration {config_name} not found")

            logger.info(f"Starting load test: {config.test_name}")

            self.is_running = True
            self.current_test = config

            # Initialize monitoring
            if config.monitor_resources:
                self._start_resource_monitoring()

            # Run test
            result = await self._execute_load_test(config)

            # Stop monitoring
            if config.monitor_resources:
                self._stop_resource_monitoring()

            # Analyze results
            result = self._analyze_test_results(result, config)

            # Store results
            self.test_results.append(result)

            self.is_running = False
            self.current_test = None

            logger.info(f"Load test completed: {config.test_name}")

            return result

        except Exception as e:
            logger.error(f"Load test failed: {e}")
            self.is_running = False
            self.current_test = None
            raise

    async def _execute_load_test(self, config: LoadTestConfig) -> LoadTestResult:
        """Execute the actual load test"""
        try:
            start_time = datetime.utcnow()

            # Initialize metrics
            total_requests = 0
            successful_requests = 0
            failed_requests = 0
            response_times = []

            # Create thread pool for concurrent users
            with ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
                # Submit tasks for each user
                futures = []

                for user_id in range(config.concurrent_users):
                    # Ramp up delay
                    ramp_delay = (config.ramp_up_time / config.concurrent_users) * user_id

                    future = executor.submit(
                        self._simulate_user,
                        config,
                        user_id,
                        ramp_delay
                    )
                    futures.append(future)

                # Wait for all tasks to complete
                user_results = []
                for future in as_completed(futures):
                    try:
                        user_result = future.result(timeout=config.test_duration + 60)
                        user_results.append(user_result)
                    except Exception as e:
                        logger.error(f"User simulation failed: {e}")

            # Aggregate results
            for user_result in user_results:
                total_requests += user_result['total_requests']
                successful_requests += user_result['successful_requests']
                failed_requests += user_result['failed_requests']
                response_times.extend(user_result['response_times'])

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            # Calculate metrics
            requests_per_second = total_requests / duration if duration > 0 else 0
            average_response_time = np.mean(response_times) if response_times else 0
            median_response_time = np.median(response_times) if response_times else 0
            p95_response_time = np.percentile(response_times, 95) if response_times else 0
            p99_response_time = np.percentile(response_times, 99) if response_times else 0
            error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0

            # Create result
            result = LoadTestResult(
                test_name=config.test_name,
                test_type=config.test_type,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                requests_per_second=requests_per_second,
                average_response_time=average_response_time,
                median_response_time=median_response_time,
                p95_response_time=p95_response_time,
                p99_response_time=p99_response_time,
                error_rate=error_rate,
                cpu_usage=self.monitoring_data['cpu'],
                memory_usage=self.monitoring_data['memory'],
                network_io=self.monitoring_data['network']
            )

            return result

        except Exception as e:
            logger.error(f"Load test execution failed: {e}")
            raise

    def _simulate_user(self, config: LoadTestConfig, user_id: int, ramp_delay: float) -> Dict[str, Any]:
        """Simulate a single user's activity"""
        try:
            # Wait for ramp-up
            time.sleep(ramp_delay)

            # Initialize user metrics
            total_requests = 0
            successful_requests = 0
            failed_requests = 0
            response_times = []

            # Calculate end time
            end_time = time.time() + config.test_duration

            # Simulate user activity
            while time.time() < end_time:
                try:
                    # Select random endpoint
                    endpoint = np.random.choice(config.target_endpoints)
                    url = f"{config.target_url}{endpoint}"

                    # Make request
                    start_time = time.time()

                    try:
                        if config.request_data:
                            response = requests.post(url, json=config.request_data, timeout=10)
                        else:
                            response = requests.get(url, timeout=10)

                        end_time_req = time.time()
                        response_time = (end_time_req - start_time) * 1000  # Convert to ms

                        total_requests += 1
                        response_times.append(response_time)

                        if response.status_code == 200:
                            successful_requests += 1
                        else:
                            failed_requests += 1

                    except requests.exceptions.RequestException:
                        failed_requests += 1
                        total_requests += 1

                    # Think time
                    time.sleep(config.think_time)

                except Exception as e:
                    logger.error(f"User {user_id} request failed: {e}")
                    failed_requests += 1
                    total_requests += 1

            return {
                'user_id': user_id,
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'response_times': response_times
            }

        except Exception as e:
            logger.error(f"User simulation failed: {e}")
            return {
                'user_id': user_id,
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'response_times': []
            }

    def _start_resource_monitoring(self):
        """Start resource monitoring"""
        try:
            def monitor_resources():
                while self.is_running:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.monitoring_data['cpu'].append(cpu_percent)

                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.monitoring_data['memory'].append(memory.percent)

                    # Network I/O
                    network = psutil.net_io_counters()
                    self.monitoring_data['network'].append({
                        'bytes_sent': network.bytes_sent,
                        'bytes_recv': network.bytes_recv,
                        'packets_sent': network.packets_sent,
                        'packets_recv': network.packets_recv
                    })

                    time.sleep(1)

            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
            monitor_thread.start()

            logger.info("Resource monitoring started")

        except Exception as e:
            logger.error(f"Resource monitoring failed: {e}")

    def _stop_resource_monitoring(self):
        """Stop resource monitoring"""
        try:
            self.is_running = False
            logger.info("Resource monitoring stopped")

        except Exception as e:
            logger.error(f"Failed to stop resource monitoring: {e}")

    def _analyze_test_results(self, result: LoadTestResult, config: LoadTestConfig) -> LoadTestResult:
        """Analyze test results and generate insights"""
        try:
            # Calculate performance score
            score = 0.0

            # RPS score (40% weight)
            rps_score = min(100, (result.requests_per_second / config.target_rps) * 100)
            score += rps_score * 0.4

            # Latency score (30% weight)
            latency_score = max(0, 100 - ((result.average_response_time / config.target_latency_ms - 1) * 100))
            score += latency_score * 0.3

            # Error rate score (30% weight)
            error_score = max(0, 100 - ((result.error_rate / config.target_error_rate - 1) * 100))
            score += error_score * 0.3

            result.performance_score = score

            # Identify bottlenecks
            bottlenecks = []

            if result.average_response_time > config.target_latency_ms * 2:
                bottlenecks.append("High response latency")

            if result.error_rate > config.target_error_rate * 2:
                bottlenecks.append("High error rate")

            if result.requests_per_second < config.target_rps * 0.5:
                bottlenecks.append("Low throughput")

            if np.mean(result.cpu_usage) > 80:
                bottlenecks.append("High CPU usage")

            if np.mean(result.memory_usage) > 80:
                bottlenecks.append("High memory usage")

            result.bottlenecks = bottlenecks

            # Generate recommendations
            recommendations = []

            if "High response latency" in bottlenecks:
                recommendations.extend([
                    "Optimize database queries",
                    "Implement caching",
                    "Add more application servers"
                ])

            if "High error rate" in bottlenecks:
                recommendations.extend([
                    "Improve error handling",
                    "Add circuit breakers",
                    "Implement retry logic"
                ])

            if "Low throughput" in bottlenecks:
                recommendations.extend([
                    "Scale horizontally",
                    "Optimize application code",
                    "Use connection pooling"
                ])

            if "High CPU usage" in bottlenecks:
                recommendations.extend([
                    "Optimize algorithms",
                    "Add more CPU resources",
                    "Implement CPU-intensive tasks in background"
                ])

            if "High memory usage" in bottlenecks:
                recommendations.extend([
                    "Optimize memory usage",
                    "Add more memory",
                    "Implement memory pooling"
                ])

            result.recommendations = recommendations

            return result

        except Exception as e:
            logger.error(f"Test result analysis failed: {e}")
            return result

    def run_scalability_test(self, base_config: str, scaling_factors: List[float]) -> Dict[str, Any]:
        """Run scalability test with different load levels"""
        try:
            logger.info(f"Starting scalability test for {base_config}")

            base_config_obj = self.test_configs.get(base_config)
            if not base_config_obj:
                raise ValueError(f"Base configuration {base_config} not found")

            scalability_results = {
                'base_config': base_config,
                'scaling_factors': scaling_factors,
                'test_results': [],
                'scaling_analysis': {}
            }

            # Run tests at different scales
            for factor in scaling_factors:
                # Create scaled configuration
                scaled_config = LoadTestConfig(
                    test_name=f"{base_config_obj.test_name} (x{factor})",
                    test_type=base_config_obj.test_type,
                    concurrent_users=int(base_config_obj.concurrent_users * factor),
                    test_duration=base_config_obj.test_duration,
                    ramp_up_time=base_config_obj.ramp_up_time,
                    think_time=base_config_obj.think_time,
                    target_url=base_config_obj.target_url,
                    target_endpoints=base_config_obj.target_endpoints,
                    request_data=base_config_obj.request_data,
                    target_rps=int(base_config_obj.target_rps * factor),
                    target_latency_ms=base_config_obj.target_latency_ms,
                    target_error_rate=base_config_obj.target_error_rate
                )

                # Run scaled test
                result = await self.run_load_test(scaled_config.test_name)
                scalability_results['test_results'].append({
                    'scaling_factor': factor,
                    'result': result
                })

                # Small delay between tests
                await asyncio.sleep(5)

            # Analyze scaling
            scalability_results['scaling_analysis'] = self._analyze_scaling(scalability_results['test_results'])

            logger.info("Scalability test completed")

            return scalability_results

        except Exception as e:
            logger.error(f"Scalability test failed: {e}")
            raise

    def _analyze_scaling(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze scaling behavior"""
        try:
            scaling_analysis = {
                'rps_scaling': {'is_linear': True, 'r_squared': 0.0, 'slope': 0.0},
                'latency_scaling': {'is_linear': True, 'r_squared': 0.0, 'slope': 0.0},
                'error_rate_scaling': {'is_linear': True, 'r_squared': 0.0, 'slope': 0.0},
                'resource_scaling': {'cpu': {}, 'memory': {}},
                'bottlenecks': [],
                'recommendations': []
            }

            # Extract data points
            factors = [r['scaling_factor'] for r in test_results]
            rps_values = [r['result'].requests_per_second for r in test_results]
            latency_values = [r['result'].average_response_time for r in test_results]
            error_rate_values = [r['result'].error_rate for r in test_results]

            # Analyze RPS scaling
            if len(factors) > 1:
                rps_slope, rps_r2 = self._calculate_linear_regression(factors, rps_values)
                scaling_analysis['rps_scaling']['slope'] = rps_slope
                scaling_analysis['rps_scaling']['r_squared'] = rps_r2
                scaling_analysis['rps_scaling']['is_linear'] = rps_r2 > 0.9

            # Analyze latency scaling
            if len(factors) > 1:
                latency_slope, latency_r2 = self._calculate_linear_regression(factors, latency_values)
                scaling_analysis['latency_scaling']['slope'] = latency_slope
                scaling_analysis['latency_scaling']['r_squared'] = latency_r2
                scaling_analysis['latency_scaling']['is_linear'] = latency_r2 > 0.9

            # Analyze error rate scaling
            if len(factors) > 1:
                error_slope, error_r2 = self._calculate_linear_regression(factors, error_rate_values)
                scaling_analysis['error_rate_scaling']['slope'] = error_slope
                scaling_analysis['error_rate_scaling']['r_squared'] = error_r2
                scaling_analysis['error_rate_scaling']['is_linear'] = error_r2 > 0.9

            # Analyze resource scaling
            for result in test_results:
                factor = result['scaling_factor']
                test_result = result['result']

                if test_result.cpu_usage:
                    avg_cpu = np.mean(test_result.cpu_usage)
                    scaling_analysis['resource_scaling']['cpu'][factor] = avg_cpu

                if test_result.memory_usage:
                    avg_memory = np.mean(test_result.memory_usage)
                    scaling_analysis['resource_scaling']['memory'][factor] = avg_memory

            # Identify bottlenecks
            if not scaling_analysis['rps_scaling']['is_linear']:
                scaling_analysis['bottlenecks'].append("Non-linear RPS scaling")

            if not scaling_analysis['latency_scaling']['is_linear']:
                scaling_analysis['bottlenecks'].append("Non-linear latency scaling")

            if scaling_analysis['resource_scaling']['cpu']:
                max_cpu = max(scaling_analysis['resource_scaling']['cpu'].values())
                if max_cpu > 80:
                    scaling_analysis['bottlenecks'].append("CPU bottleneck at scale")

            if scaling_analysis['resource_scaling']['memory']:
                max_memory = max(scaling_analysis['resource_scaling']['memory'].values())
                if max_memory > 80:
                    scaling_analysis['bottlenecks'].append("Memory bottleneck at scale")

            # Generate recommendations
            if scaling_analysis['bottlenecks']:
                scaling_analysis['recommendations'].extend([
                    "Investigate non-linear scaling behavior",
                    "Optimize resource utilization",
                    "Consider horizontal scaling",
                    "Implement auto-scaling policies"
                ])
            else:
                scaling_analysis['recommendations'].append("System scales linearly - ready for production")

            return scaling_analysis

        except Exception as e:
            logger.error(f"Scaling analysis failed: {e}")
            return {}

    def _calculate_linear_regression(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        """Calculate linear regression slope and R-squared"""
        try:
            if len(x) < 2 or len(y) < 2:
                return 0.0, 0.0

            x_array = np.array(x)
            y_array = np.array(y)

            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, y_array)

            return slope, r_value ** 2

        except Exception as e:
            logger.error(f"Linear regression calculation failed: {e}")
            return 0.0, 0.0

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            report = {
                'report_id': hashlib.sha256(f"report_{time.time()}".encode()).hexdigest()[:16],
                'generated_at': datetime.utcnow(),
                'test_summary': {},
                'performance_trends': {},
                'bottleneck_analysis': {},
                'capacity_analysis': {},
                'recommendations': []
            }

            if not self.test_results:
                report['test_summary'] = {'message': 'No test results available'}
                return report

            # Test summary
            report['test_summary'] = {
                'total_tests': len(self.test_results),
                'average_performance_score': np.mean([r.performance_score for r in self.test_results]),
                'best_performing_test': max(self.test_results, key=lambda r: r.performance_score).test_name,
                'worst_performing_test': min(self.test_results, key=lambda r: r.performance_score).test_name
            }

            # Performance trends
            report['performance_trends'] = {
                'rps_trend': [r.requests_per_second for r in self.test_results],
                'latency_trend': [r.average_response_time for r in self.test_results],
                'error_rate_trend': [r.error_rate for r in self.test_results],
                'performance_score_trend': [r.performance_score for r in self.test_results]
            }

            # Bottleneck analysis
            all_bottlenecks = []
            for result in self.test_results:
                all_bottlenecks.extend(result.bottlenecks)

            bottleneck_counts = {}
            for bottleneck in all_bottlenecks:
                bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1

            report['bottleneck_analysis'] = {
                'top_bottlenecks': sorted(bottleneck_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                'total_bottlenecks': len(all_bottlenecks)
            }

            # Capacity analysis
            max_rps = max([r.requests_per_second for r in self.test_results])
            max_concurrent_users = max([r.total_requests for r in self.test_results])

            report['capacity_analysis'] = {
                'max_throughput_rps': max_rps,
                'max_concurrent_users': max_concurrent_users,
                'estimated_capacity': {
                    'daily_requests': max_rps * 86400,
                    'monthly_requests': max_rps * 2592000,
                    'yearly_requests': max_rps * 31536000
                }
            }

            # Recommendations
            all_recommendations = []
            for result in self.test_results:
                all_recommendations.extend(result.recommendations)

            # Remove duplicates and prioritize
            unique_recommendations = list(set(all_recommendations))
            report['recommendations'] = unique_recommendations[:10]

            return report

        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return {'error': str(e)}

    def get_load_testing_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive load testing dashboard"""
        try:
            dashboard = {
                'overview': {
                    'total_tests': len(self.test_results),
                    'tests_running': 1 if self.is_running else 0,
                    'last_test': self.test_results[-1].test_name if self.test_results else None
                },
                'performance_metrics': {},
                'resource_utilization': {},
                'test_results': [],
                'scaling_metrics': {}
            }

            # Performance metrics
            if self.test_results:
                latest_result = self.test_results[-1]
                dashboard['performance_metrics'] = {
                    'rps': latest_result.requests_per_second,
                    'average_latency': latest_result.average_response_time,
                    'p95_latency': latest_result.p95_response_time,
                    'error_rate': latest_result.error_rate,
                    'performance_score': latest_result.performance_score
                }

            # Resource utilization
            if self.monitoring_data['cpu']:
                dashboard['resource_utilization'] = {
                    'current_cpu': self.monitoring_data['cpu'][-1] if self.monitoring_data['cpu'] else 0,
                    'current_memory': self.monitoring_data['memory'][-1] if self.monitoring_data['memory'] else 0,
                    'avg_cpu': np.mean(self.monitoring_data['cpu']) if self.monitoring_data['cpu'] else 0,
                    'avg_memory': np.mean(self.monitoring_data['memory']) if self.monitoring_data['memory'] else 0
                }

            # Recent test results
            dashboard['test_results'] = [
                {
                    'test_name': result.test_name,
                    'test_type': result.test_type,
                    'performance_score': result.performance_score,
                    'rps': result.requests_per_second,
                    'latency': result.average_response_time,
                    'error_rate': result.error_rate,
                    'timestamp': result.start_time.isoformat()
                }
                for result in self.test_results[-10:]
            ]

            return dashboard

        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            return {'error': str(e)}


# Global load testing instance
_load_testing_framework = None

def get_load_testing_framework() -> LoadTestingFramework:
    """Get global load testing framework instance"""
    global _load_testing_framework
    if _load_testing_framework is None:
        _load_testing_framework = LoadTestingFramework()
    return _load_testing_framework


async def main():
    # Test load testing framework
    framework = LoadTestingFramework()

    # Run a simple load test
    print("Running load test...")
    result = await framework.run_load_test('trading_engine_load')
    print(f"Load test result: {json.dumps(result.__dict__, indent=2, default=str)}")

    # Generate performance report
    report = framework.generate_performance_report()
    print(f"Performance report: {json.dumps(report, indent=2, default=str)}")

    # Get dashboard
    dashboard = framework.get_load_testing_dashboard()
    print(f"Dashboard: {json.dumps(dashboard, indent=2, default=str)}")

if __name__ == "__main__":
    asyncio.run(main())

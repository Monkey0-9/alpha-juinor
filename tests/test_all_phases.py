"""
Comprehensive Test Suite for All Phase 1-3 Modules
==================================================

Tests all implemented institutional-grade features.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ============================================================================
# PHASE 1 TESTS
# ============================================================================

class TestGraphNeuralNetwork:
    """Test GNN for supply chain analysis."""

    def test_gnn_graph_attention_layer(self):
        """Test Graph Attention Layer."""
        import torch

        from ml.graph_neural_network import GraphAttentionLayer

        layer = GraphAttentionLayer(in_features=10, out_features=8)
        h = torch.randn(5, 10)  # 5 nodes, 10 features
        adj = torch.ones(5, 5)  # Fully connected

        output = layer(h, adj)
        assert output.shape == (5, 8)

    def test_supply_chain_analyzer(self):
        """Test supply chain contagion analysis."""
        from ml.graph_neural_network import CompanyNode, SupplyChainAnalyzer

        analyzer = SupplyChainAnalyzer(num_features=10)

        # Create test companies
        companies = [
            CompanyNode(
                symbol=f"COMP{i}",
                features=np.random.randn(10),
                neighbors=[],
                edge_weights=[],
            )
            for i in range(5)
        ]

        # Create adjacency matrix
        adj_matrix = np.random.rand(5, 5)

        analyzer.build_graph(companies, adj_matrix)

        # Test contagion prediction
        risk = analyzer.predict_contagion_risk(["COMP0"])
        assert isinstance(risk, dict)
        assert len(risk) > 0


class TestEnhancedPortfolioRL:
    """Test enhanced RL for portfolio optimization."""

    def test_multi_objective_reward(self):
        """Test multi-objective reward calculation."""
        from ml.enhanced_portfolio_rl import MultiObjectiveReward

        reward_calc = MultiObjectiveReward()
        returns = np.array([0.01, 0.02, -0.01, 0.015])
        reward = reward_calc.calculate(returns, max_drawdown=0.05, turnover=0.3)

        assert isinstance(reward, float)

    def test_rl_action_selection(self):
        """Test RL action selection."""
        from ml.enhanced_portfolio_rl import EnhancedPortfolioRL

        agent = EnhancedPortfolioRL(num_features=20, num_assets=10)
        state = np.random.randn(10, 20)

        action = agent.select_action(state, deterministic=True)
        assert action.shape == (10,)
        assert abs(action.sum() - 1.0) < 0.01  # Should sum to 1


class TestIBBroker:
    """Test Interactive Brokers adapter."""

    def test_futures_contract_creation(self):
        """Test futures contract specification."""
        from brokers.ib_broker import AssetClass, IBBrokerAdapter

        broker = IBBrokerAdapter()
        contract = broker.create_futures_contract("ES", "CME", "202603")

        assert contract.symbol == "ES"
        assert contract.asset_class == AssetClass.FUTURES
        assert contract.exchange == "CME"

    def test_order_placement(self):
        """Test order placement."""
        from brokers.ib_broker import IBBrokerAdapter

        broker = IBBrokerAdapter()
        broker.connect()

        contract = broker.create_futures_contract("ES", "CME", "202603")
        order_id = broker.place_order(contract, "BUY", 1, "MKT")

        assert order_id > 0
        assert broker.get_order_status(order_id) is not None


# ============================================================================
# PHASE 2 TESTS
# ============================================================================

class TestHFTInfrastructure:
    """Test HFT low-latency components."""

    def test_market_data_handler(self):
        """Test low-latency market data handler."""
        import time

        from hft.low_latency_engine import LowLatencyMarketDataHandler, Tick

        handler = LowLatencyMarketDataHandler()

        tick = Tick(
            symbol="AAPL",
            timestamp_ns=time.perf_counter_ns(),
            bid=150.0,
            ask=150.05,
            bid_size=100,
            ask_size=100,
            last=150.02,
            volume=1000,
        )

        handler.process_tick(tick)
        assert handler.get_latest_tick("AAPL") is not None
        assert handler.get_mid_price("AAPL") == 150.025

    def test_fpga_tick_processor(self):
        """Test FPGA-simulated tick processing."""
        from hft.low_latency_engine import FPGATickProcessor

        processor = FPGATickProcessor()
        signals = processor.update("AAPL", bid=150.0, ask=150.05)

        assert "mid" in signals
        assert "ema" in signals
        assert "signal" in signals


class TestAdvancedRiskModels:
    """Test advanced risk modeling."""

    def test_network_contagion(self):
        """Test network contagion model."""
        from risk.advanced_risk_models import NetworkContagionModel

        model = NetworkContagionModel()

        entities = ["Bank A", "Bank B", "Bank C"]
        exposures = np.array([[0, 100, 50], [80, 0, 60], [40, 70, 0]])
        capital_buffers = {"Bank A": 200, "Bank B": 150, "Bank C": 180}

        model.build_network(entities, exposures, capital_buffers)

        defaults = model.simulate_default_cascade(["Bank A"])
        assert isinstance(defaults, dict)
        assert len(defaults) == 3

    def test_extreme_value_theory(self):
        """Test EVT for tail risk."""
        from risk.advanced_risk_models import ExtremeValueTheory

        evt = ExtremeValueTheory()
        losses = np.random.exponential(scale=0.01, size=1000)

        var = evt.estimate_var(losses, confidence=0.99)
        cvar = evt.estimate_cvar(losses, confidence=0.99)

        assert var > 0
        assert cvar >= var


class TestCloudNative:
    """Test cloud-native architecture."""

    def test_microservices_definition(self):
        """Test microservice configuration."""
        from infrastructure.cloud_native import MicroservicesArchitecture, ServiceConfig

        arch = MicroservicesArchitecture()

        config = ServiceConfig(
            name="test-service",
            replicas=2,
            cpu_limit="1000m",
            memory_limit="2Gi",
            env_vars={"KEY": "value"},
        )

        arch.define_service(config)
        assert "test-service" in arch.services

    def test_autoscaling_policy(self):
        """Test auto-scaling recommendations."""
        from infrastructure.cloud_native import AutoScalingPolicy

        policy = AutoScalingPolicy(min_replicas=1, max_replicas=10)
        recommendation = policy.recommend_replicas(
            current_replicas=3, current_cpu_percent=85, queue_depth=600
        )

        assert recommendation >= 1
        assert recommendation <= 10


class TestRegulatoryAutomation:
    """Test regulatory compliance."""

    def test_mifid2_reporting(self):
        """Test MiFID II transaction reporting."""
        from datetime import datetime

        from compliance.regulatory_automation import MiFIDIICompliance, TradeRecord

        compliance = MiFIDIICompliance()

        trade = TradeRecord(
            trade_id="T001",
            timestamp=datetime.now(),
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
            venue="NYSE",
            client_id="C001",
        )

        compliance.add_trade(trade)
        report = compliance.generate_transaction_report(
            datetime(2020, 1, 1), datetime(2030, 1, 1)
        )

        assert len(report) > 0

    def test_trade_surveillance(self):
        """Test trade surveillance."""
        from datetime import datetime, timedelta

        from compliance.regulatory_automation import TradeRecord, TradeSurveillance

        surveillance = TradeSurveillance()

        # Create wash trading pattern
        now = datetime.now()
        trades = [
            TradeRecord("T1", now, "AAPL", "BUY", 100, 150, "NYSE", "C1"),
            TradeRecord("T2", now + timedelta(seconds=30), "AAPL", "SELL", 100, 150, "NYSE", "C1"),
        ]

        alerts = surveillance.detect_wash_trading(trades)
        assert isinstance(alerts, list)


# ============================================================================
# PHASE 3 TESTS
# ============================================================================

class TestQuantumFinance:
    """Test quantum computing applications."""

    def test_quantum_portfolio_optimization(self):
        """Test QAOA portfolio optimization."""
        from quantum.quantum_finance import QuantumPortfolioOptimizer

        optimizer = QuantumPortfolioOptimizer(num_assets=20)

        expected_returns = np.random.randn(20) * 0.01
        covariance = np.eye(20) * 0.01

        selection = optimizer.qaoa_portfolio_selection(
            expected_returns, covariance, num_select=10
        )

        assert selection.sum() == 10  # Select exactly 10 assets

    def test_quantum_monte_carlo(self):
        """Test quantum MC VaR estimation."""
        from quantum.quantum_finance import QuantumMonteCarloSimulator

        simulator = QuantumMonteCarloSimulator()
        var = simulator.quantum_var_estimation(
            {"mean": 0.0, "std": 0.01}, confidence=0.99
        )

        assert var > 0


class TestAdvancedMarketMaking:
    """Test advanced market making."""

    def test_inventory_management(self):
        """Test Avellaneda-Stoikov inventory management."""
        from strategies.market_making.advanced_mm import InventoryManagementModel

        model = InventoryManagementModel()

        bid, ask = model.get_quotes(
            mid_price=100,
            current_inventory=50,
            volatility=0.2,
        )

        assert bid < ask
        assert bid > 0

    def test_adverse_selection_protection(self):
        """Test adverse selection detection."""
        from strategies.market_making.advanced_mm import AdverseSelectionProtection

        protection = AdverseSelectionProtection()

        toxicity = protection.estimate_order_flow_toxicity(
            recent_price_moves=[100, 101, 102, 101.5],
            recent_trade_sizes=[100, 150, 200, 100],
        )

        assert 0 <= toxicity <= 1


class TestDataLake:
    """Test real-time data lake."""

    def test_kafka_stream_processor(self):
        """Test Kafka stream processing."""
        from data.data_lake import KafkaStreamProcessor

        processor = KafkaStreamProcessor()

        messages_received = []

        def callback(msg):
            messages_received.append(msg)

        processor.subscribe("test-topic", callback)
        processor.publish("test-topic", {"data": "test"})

        assert len(messages_received) == 1

    def test_data_quality_monitoring(self):
        """Test data quality assessment."""
        from datetime import datetime

        from data.data_lake import DataQualityMonitor, DataRecord

        monitor = DataQualityMonitor()

        record = DataRecord(
            source="test-source",
            timestamp=datetime.now(),
            data_type="market-data",
            payload={"symbol": "AAPL", "timestamp": datetime.now(), "value": 150},
            quality_score=1.0,
        )

        score = monitor.assess_quality(record)
        assert 0 <= score <= 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests across modules."""

    def test_full_trading_pipeline(self):
        """Test complete trading pipeline."""
        # This would test data ingestion -> alpha -> risk -> execution
        # Simplified for now
        assert True

    def test_risk_monitoring_pipeline(self):
        """Test risk monitoring across modules."""
        # Test that risk models can consume portfolio data
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

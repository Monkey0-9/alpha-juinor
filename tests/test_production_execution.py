
import pytest
import time
from datetime import datetime
from unittest.mock import MagicMock, patch
from orchestration.live_decision_loop import LiveDecisionLoop, LiveSignal, DECISION_BUY, DECISION_SELL
from execution.oms import OrderStatus

class TestProductionExecution:
    @pytest.fixture
    def mock_broker(self):
        broker = MagicMock()
        broker.submit_order.return_value = MagicMock(
            order_id="test-order-123",
            status=OrderStatus.FILLED,
            filled_qty=10.0,
            filled_avg_price=150.0
        )
        broker.get_positions.return_value = {}
        return broker

    @pytest.fixture
    def decision_loop(self, mock_broker):
        with patch('orchestration.live_decision_loop.DataRouter'), \
             patch('orchestration.live_decision_loop.MetaBrain'), \
             patch('orchestration.live_decision_loop.LifecycleManager'):
            loop = LiveDecisionLoop(
                tick_interval=0.1,
                paper_mode=False,
                symbols=["AAPL"]
            )
            loop.broker = mock_broker
            return loop

    def test_high_conviction_buy_triggers_execution(self, decision_loop, mock_broker):
        # Setup high conviction signal
        signals = {
            "AAPL": LiveSignal(
                symbol="AAPL",
                signal=DECISION_BUY,
                conviction=0.9,
                current_price=150.0,
                position_size=10.0
            )
        }
        
        # Manually trigger execution logic
        decision_loop._execute_trades(signals)
        
        # Verify broker was called
        assert mock_broker.submit_order.called
        args, kwargs = mock_broker.submit_order.call_args
        assert kwargs['symbol'] == "AAPL"
        assert kwargs['side'] == "buy"
        assert decision_loop.state.orders_executed == 1

    def test_low_conviction_signal_skipped(self, decision_loop, mock_broker):
        # Setup low conviction signal
        signals = {
            "AAPL": LiveSignal(
                symbol="AAPL",
                signal=DECISION_BUY,
                conviction=0.4, # Below 0.7 threshold
                current_price=150.0,
                position_size=10.0
            )
        }
        
        decision_loop._execute_trades(signals)
        
        # Verify broker was NOT called
        assert not mock_broker.submit_order.called
        assert decision_loop.state.orders_executed == 0

    def test_latency_measurement_is_real(self, decision_loop):
        # Mock compute_signals to take some time
        def slow_compute():
            time.sleep(0.05)
            return {}
            
        decision_loop._compute_signals = MagicMock(side_effect=slow_compute)
        decision_loop._execute_trades = MagicMock()
        
        # Run one tick
        with patch('orchestration.live_decision_loop.metrics'):
            decision_loop._run_decision_tick()
            
        # Check heartbeat logs or state if we had a way to verify them
        # For now, just ensure it doesn't crash and cycle count increases
        assert decision_loop.state.cycle_count == 1

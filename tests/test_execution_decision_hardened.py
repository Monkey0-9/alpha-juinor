
import pytest
from governance.execution_decision import decide_execution

@pytest.fixture
def default_config():
    return {
        'trading_mode': 'cautious',
        'execution': {
            'min_notional_usd': 200,
            'min_weight_change': 0.0025,
            'min_conviction': 0.30,
            'max_skip_cycles': 3,
            'min_qty': 1
        }
    }

def test_skip_not_tradable_yahoo_placeholder(default_config):
    decision = decide_execution(
        cycle_id='test',
        symbol='EURUSD=X',
        target_weight=0.05,
        current_weight=0.0,
        nav_usd=100000,
        price=1.1,
        conviction=0.8,
        data_quality=1.0,
        risk_scaled_weight=0.05,
        skipping_history={},
        market_open=True,
        config=default_config
    )
    assert decision['decision'] == 'SKIP_NOT_TRADABLE'
    assert 'SYMBOL_UNSUPPORTED' in decision['reason_codes']

def test_skip_not_tradable_index(default_config):
    decision = decide_execution(
        cycle_id='test',
        symbol='^GSPC',
        target_weight=0.05,
        current_weight=0.0,
        nav_usd=100000,
        price=4000,
        conviction=0.8,
        data_quality=1.0,
        risk_scaled_weight=0.05,
        skipping_history={},
        market_open=True,
        config=default_config
    )
    assert decision['decision'] == 'SKIP_NOT_TRADABLE'
    assert 'INDEX_NOT_TRADABLE' in decision['reason_codes']

def test_min_qty_check(default_config):
    # Price high enough to meet notional ($200), but qty < 1?
    # e.g. BRK.A price $500,000. Notional $200. Qty 0.0004. Rounded 0.
    # Default min_qty is 1.

    decision = decide_execution(
        cycle_id='test',
        symbol='HIGHPRICE',
        target_weight=0.001, # $100 notional. Too small anyway?
        # Let's make notional ok but qty small.
        # Notional > $200. Price $500,000. Qty -> 0.0004 = 0.
        current_weight=0.0,
        nav_usd=1000000, # 1M NAV
        price=500000.0,
        # Target weight 0.0003 -> $300.
        conviction=0.8,
        data_quality=1.0,
        risk_scaled_weight=0.0003,
        skipping_history={},
        market_open=True,
        config=default_config
    )
    # Target Notional $300 > $200.
    # Weight delta 0.0003 < 0.0025. It will skip due to weight delta.
    # Need to pass weight delta too.

    # Let's force delta pass.
    # Case: Holding 0. Weight Change > 0.0025.
    # Target 0.0025 -> $2500.
    # Price $500,000. Qty 0.005 -> 0.

    decision = decide_execution(
        cycle_id='test',
        symbol='HIGHPRICE',
        target_weight=0.0026,
        current_weight=0.0,
        nav_usd=1000000,
        price=500000.0,
        conviction=0.8,
        data_quality=1.0,
        risk_scaled_weight=0.0026,
        skipping_history={},
        market_open=True,
        config=default_config
    )

    # Weight delta 0.0026 > 0.0025. OK.
    # Notional $2600 > $200. OK.
    # Qty 0.0052 -> 0.
    # Should SKIP due to QTY if min_qty=1

    # Wait, my logic checks min_qty if precision is 0.
    # decision['decision'] should be SKIP_TOO_SMALL or some such.

    # Based on logic:
    # rounded_target_qty = 0.
    # if rounded_target_qty < min_qty (1) ... small_reasons.append('QTY_TOO_SMALL')
    # returns SKIP_TOO_SMALL.

    assert decision['decision'] == 'SKIP_TOO_SMALL'
    assert 'QTY_TOO_SMALL' in decision['reason_codes'] or 'ORDER_QTY_ZERO' in decision['reason_codes']

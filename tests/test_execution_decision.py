
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
            'max_skip_cycles': 3
        }
    }

def test_execute_happy_path(default_config):
    decision = decide_execution(
        cycle_id='test_cycle',
        symbol='AAPL',
        target_weight=0.05, # 5%
        current_weight=0.0,
        nav_usd=100000,
        price=150.0,
        conviction=0.8,
        data_quality=1.0,
        risk_scaled_weight=0.05,
        skipping_history={},
        market_open=True,
        config=default_config
    )

    assert decision['decision'] == 'EXECUTE'
    # Target Notional = 0.05 * 100000 = 5000 > 200
    # Weight Delta = 0.05 > 0.0025
    # Conviction 0.8 > 0.3

def test_skip_too_small_notional(default_config):
    decision = decide_execution(
        cycle_id='test_cycle',
        symbol='AAPL',
        target_weight=0.001, # 0.1% of 100k = $100 < $200
        current_weight=0.0,
        nav_usd=100000,
        price=150.0,
        conviction=0.8,
        data_quality=1.0,
        risk_scaled_weight=0.001,
        skipping_history={},
        market_open=True,
        config=default_config
    )

    assert decision['decision'] == 'SKIP_TOO_SMALL'
    assert 'NOTIONAL_TOO_SMALL' in decision['reason_codes']

def test_skip_too_small_weight_delta(default_config):
    decision = decide_execution(
        cycle_id='test_cycle',
        symbol='AAPL',
        target_weight=0.051,
        current_weight=0.050, # Delta 0.001 < 0.0025
        nav_usd=100000,
        price=150.0,
        conviction=0.8,
        data_quality=1.0,
        risk_scaled_weight=0.051,
        skipping_history={},
        market_open=True,
        config=default_config
    )

    assert decision['decision'] == 'SKIP_TOO_SMALL'
    assert 'WEIGHT_DELTA_TOO_SMALL' in decision['reason_codes']

def test_skip_low_conviction(default_config):
    decision = decide_execution(
        cycle_id='test_cycle',
        symbol='AAPL',
        target_weight=0.05,
        current_weight=0.0,
        nav_usd=100000,
        price=150.0,
        conviction=0.1, # < 0.3
        data_quality=1.0,
        risk_scaled_weight=0.05,
        skipping_history={},
        market_open=True,
        config=default_config
    )

    assert decision['decision'] == 'SKIP_LOW_CONFIDENCE'
    assert 'LOW_CONVICTION' in decision['reason_codes']

def test_market_closed(default_config):
    decision = decide_execution(
        cycle_id='test_cycle',
        symbol='AAPL',
        target_weight=0.05,
        current_weight=0.0,
        nav_usd=100000,
        price=150.0,
        conviction=0.8,
        data_quality=1.0,
        risk_scaled_weight=0.05,
        skipping_history={},
        market_open=False, # Closed
        config=default_config
    )

    assert decision['decision'] == 'SKIP_MARKET_CLOSED'

def test_force_execute_active_mode(default_config):
    # Enable active mode
    active_config = default_config.copy()
    active_config['trading_mode'] = 'active'

    # 3 prior skips -> should trigger force
    skipping_history = {'AAPL': 3}

    decision = decide_execution(
        cycle_id='test_cycle',
        symbol='AAPL',
        target_weight=0.0005, # Tiny, normally skip
        current_weight=0.0,
        nav_usd=100000,
        price=10.0, # Notional $50 < $100 (active min)
        conviction=0.8,
        data_quality=1.0,
        risk_scaled_weight=0.0005,
        skipping_history=skipping_history,
        market_open=True,
        config=active_config
    )

    # Active mode min notional is 100. Target is 50. Should SKIP_TOO_SMALL.
    # But Max Skip Cycles = 3 is met.
    # Expect EXECUTE with FORCING reason.
    assert decision['decision'] == 'EXECUTE'
    assert 'EXECUTE_FORCING_MAX_SKIP_CYCLES' in decision['reason_codes']

def test_no_force_execute_research_mode(default_config):
    # Research mode -> NO forcing allowed (per prompt "Optional: only force execute in active or aggressive")
    research_config = default_config.copy()
    research_config['trading_mode'] = 'research'

    skipping_history = {'AAPL': 10} # Many skips

    decision = decide_execution(
        cycle_id='test_cycle',
        symbol='AAPL',
        target_weight=0.0001,
        current_weight=0.0,
        nav_usd=100000,
        price=10.0,
        conviction=0.8,
        data_quality=1.0,
        risk_scaled_weight=0.0001,
        skipping_history=skipping_history,
        market_open=True,
        config=research_config
    )

    assert decision['decision'] == 'SKIP_TOO_SMALL'
    assert 'EXECUTE_FORCING_MAX_SKIP_CYCLES' not in decision['reason_codes']

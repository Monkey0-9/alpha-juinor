
"""
Execution Decision Layer.
Core logic for deciding whether to EXECUTE or SKIP a trade based on risk, conviction, and operational constraints.
"""

import math
from typing import Dict, List, Any, Optional
from governance.explainer import DecisionExplainer

explainer = DecisionExplainer()

def decide_execution(
    cycle_id: str,
    symbol: str,
    target_weight: float,
    current_weight: float,
    nav_usd: float,
    price: float,
    conviction: float,
    data_quality: float,
    risk_scaled_weight: float,
    skipping_history: Dict[str, int],
    market_open: bool,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Decides whether to execute a trade for a single symbol.
    Strictly enforces institutional constraints.

    Args:
        cycle_id: ID of the current cycle.
        symbol: Ticker symbol.
        target_weight: Desired portfolio weight (signed).
        current_weight: Current portfolio weight (signed).
        nav_usd: Net Asset Value in USD.
        price: Current asset price.
        conviction: Signal conviction [0, 1].
        data_quality: Data quality score [0, 1].
        risk_scaled_weight: Target weight after risk scaling.
        skipping_history: Dict mapping symbol -> count of consecutive skipped cycles.
        market_open: Boolean indicating if market is open.
        config: Configuration dictionary.

    Returns:
        Dict containing decision, reason_codes, target_qty, notional_usd, rounded_qty, etc.
    """

    # 0. Input Validation - Catch NaN/Inf early
    if not math.isfinite(target_weight) or not math.isfinite(current_weight):
        return _build_decision(
            cycle_id, symbol, 'SKIP_INVALID_DATA',
            ['INVALID_WEIGHT_INPUT'],
            0.0, 0.0, nav_usd, price, conviction, 0.0, 0
        )

    if not math.isfinite(nav_usd) or nav_usd <= 0:
        return _build_decision(
            cycle_id, symbol, 'SKIP_INVALID_DATA',
            ['INVALID_NAV'],
            target_weight, current_weight, 0.0, price, conviction, 0.0, 0
        )

    if not math.isfinite(price) or price <= 0:
        return _build_decision(
            cycle_id, symbol, 'SKIP_INVALID_DATA',
            ['INVALID_PRICE'],
            target_weight, current_weight, nav_usd, 0.0, conviction, 0.0, 0
        )

    # 1. Resolve Configuration based on Trading Mode
    trading_mode = config.get('trading_mode', 'cautious')
    exec_config = config.get('execution', {})

    # Defaults
    min_notional_usd = exec_config.get('min_notional_usd', 200)
    # Support both key names for weight change
    min_weight_change = exec_config.get('min_weight_change_pct', exec_config.get('min_weight_change', 0.0025))
    min_conviction = exec_config.get('min_conviction', 0.30)
    max_skip_cycles = exec_config.get('max_skip_cycles', 3)
    min_qty = exec_config.get('min_qty', 1)

    # Apply Mode Overrides
    if trading_mode == 'research':
        min_conviction = 0.6
        min_notional_usd = 1000
    elif trading_mode == 'cautious': # Default
        min_conviction = 0.30
        min_notional_usd = 200
    elif trading_mode == 'active':
        min_conviction = 0.20
        min_notional_usd = 100
    elif trading_mode == 'aggressive':
        min_conviction = 0.10
        min_notional_usd = 50

    reasons: List[str] = []

    # Calculate notional and qty
    weight_delta = target_weight - current_weight

    # Decide if we are opening, closing, or adjusting
    # For execution decision, we care about the ORDER size (Delta).
    # But for "Too Small to Hold", we care about Target.

    target_notional_usd = abs(target_weight) * nav_usd
    order_notional_usd = abs(weight_delta) * nav_usd

    # Calculate Qty based on Target Weight (Target Allocation)
    raw_target_qty = target_notional_usd / (price if price > 0 else 1.0)

    # Calculate Order Qty (Delta)
    raw_order_qty = order_notional_usd / (price if price > 0 else 1.0)

    # Validate quantities before rounding (NaN/Inf protection)
    if not math.isfinite(raw_target_qty) or not math.isfinite(raw_order_qty):
        return _build_decision(
            cycle_id, symbol, 'SKIP_INVALID_DATA',
            ['INVALID_QTY_CALCULATION'],
            target_weight, current_weight, nav_usd, price, conviction,
            0.0, 0
        )

    # Rounding logic
    qty_precision = 0 # Default for stocks
    if 'USD' in symbol and '=' not in symbol: # Crypto pair guess?
        qty_precision = 4

    if qty_precision == 0:
        rounded_target_qty = round(raw_target_qty)
        rounded_order_qty = round(raw_order_qty)
    else:
        rounded_target_qty = round(raw_target_qty, qty_precision)
        rounded_order_qty = round(raw_order_qty, qty_precision)

    # 0. Asset Class / Tradability Check (Symbol Gate)
    # Reject symbols containing =X, =F (Yahoo placeholders for FX/Futures), or known unsupported assets.
    if '=X' in symbol or '=F' in symbol:
        return _build_decision(cycle_id, symbol, 'SKIP_NOT_TRADABLE', ['SYMBOL_UNSUPPORTED'], target_weight, current_weight, nav_usd, price, conviction, target_notional_usd, rounded_target_qty)

    # Additional Yahoo-style checks if necessary
    if '^' in symbol: # Yahoo indexes
        return _build_decision(cycle_id, symbol, 'SKIP_NOT_TRADABLE', ['INDEX_NOT_TRADABLE'], target_weight, current_weight, nav_usd, price, conviction, target_notional_usd, rounded_target_qty)

    # 2. Market Status Check
    if not market_open:
        return _build_decision(cycle_id, symbol, 'SKIP_MARKET_CLOSED', ['MARKET_CLOSED'], target_weight, current_weight, nav_usd, price, conviction, target_notional_usd, rounded_target_qty)

    # 3. Risk Scaled Zero Check
    if abs(risk_scaled_weight) < 1e-8 and abs(target_weight) > 1e-8:
        # Risk engine reduced it to zero, but we wanted a position
        return _build_decision(cycle_id, symbol, 'SKIP_RISK_ZERO', ['RISK_SCALED_ZERO'], target_weight, current_weight, nav_usd, price, conviction, target_notional_usd, rounded_target_qty)

    # If target is actually 0 (we want to close), we shouldn't skip due to risk zero.

    # 4. Weight Delta Check (Turnover control)
    # Only skip if the CHANGE is small. If we are closing a position (target=0),
    # and current pos is large, change is large. If current pos is tiny, change is tiny.
    should_skip_small_delta = False

    if abs(weight_delta) < min_weight_change:
        # Check if we are holding a tiny position and trying to close it.
        # If target = 0 and current < min_weight_change:
        # We allow closing dust if we are running in active mode or manually flagged.
        # But generally, avoid trading dust.
        should_skip_small_delta = True

    # 5. Notional & Qty Check
    should_skip_small_size = False
    small_reasons = []

    if should_skip_small_delta:
        small_reasons.append('WEIGHT_DELTA_TOO_SMALL')
        should_skip_small_size = True

    # "If target notional or rounded qty < MIN_NOTIONAL_USD"
    # This check is primarily about "Is this position worth holding?".
    # If target is 0, we are not holding it.
    if abs(target_weight) > 1e-8:
        if target_notional_usd < min_notional_usd:
             should_skip_small_size = True
             small_reasons.append('NOTIONAL_TOO_SMALL')

        if rounded_target_qty < min_qty and qty_precision == 0:
             # Only apply min_qty check for integer assets (stocks)
             should_skip_small_size = True
             small_reasons.append('QTY_TOO_SMALL')

    # Force Execution Check (Max Skip Cycles)
    is_forced = False
    if should_skip_small_size:
        # If we are skipping, check if we should force
        skip_count = skipping_history.get(symbol, 0)
        force_allowed = trading_mode in ['active', 'aggressive']

        if force_allowed and skip_count >= max_skip_cycles:
             # Only force if it's a valid tradeable asset, just small.
             is_forced = True
             reasons.append('EXECUTE_FORCING_MAX_SKIP_CYCLES')
        else:
             return _build_decision(cycle_id, symbol, 'SKIP_TOO_SMALL', small_reasons, target_weight, current_weight, nav_usd, price, conviction, target_notional_usd, rounded_target_qty)

    # 6. Conviction Check
    if conviction < min_conviction and not is_forced:
        return _build_decision(cycle_id, symbol, 'SKIP_LOW_CONFIDENCE', ['LOW_CONVICTION'], target_weight, current_weight, nav_usd, price, conviction, target_notional_usd, rounded_target_qty)

    # 7. Check if Order Quantity is 0 (after rounding)
    # Even if everything else passed, if we round to 0 shares to trade, we can't trade.
    # Note: rounded_order_qty is the delta.
    if rounded_order_qty == 0:
        return _build_decision(cycle_id, symbol, 'SKIP_TOO_SMALL', ['ORDER_QTY_ZERO'], target_weight, current_weight, nav_usd, price, conviction, target_notional_usd, rounded_target_qty)

    # 8. All checks passed
    final_decision = 'EXECUTE'
    if not reasons: reasons = ['OK'] # Default reason

    # --- Explainability Integration ---
    # Construct explanation
    risk_metrics = {
        "nav_usd": (nav_usd, 0, ">") # Basic check
    }

    explanation_obj = explainer.explain_trade(
        symbol=symbol,
        action=final_decision,
        signal_strength=conviction * 3.0, # Approximate sigma from conviction (0-1 -> 0-3)
        signal_components={}, # Metadata missing in current signature
        position_size=rounded_order_qty,
        adv=1e6, # Placeholder, real ADV usually passed in config or lookup
        risk_metrics=risk_metrics,
        governance_approved=True,
        confidence_interval=explainer.compute_confidence_bands(conviction * 3.0, 0.05)
    )

    explanation_text = explainer.format_explanation(explanation_obj)

    return _build_decision(cycle_id, symbol, final_decision, reasons, target_weight, current_weight, nav_usd, price, conviction, target_notional_usd, rounded_target_qty, explanation=explanation_text)

def _build_decision(cycle_id, symbol, decision, reason_codes, target_weight, current_weight, nav_usd, price, conviction, notional_usd, rounded_qty, explanation=None):
    return {
       'decision': decision,
       'reason_codes': reason_codes,
       'target_qty': rounded_qty,  # Position Qty
       'notional_usd': notional_usd,
       'target_weight': target_weight,
       'current_weight': current_weight,
       'conviction': conviction,
       'symbol': symbol,
       'cycle_id': cycle_id,
       'explanation': explanation
    }

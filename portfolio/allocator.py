from portfolio.capital_auction import CapitalAuctionEngine
from contracts import AllocationRequest
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import logging

# Institutional Infrastructure (Phase 8 Integration)
from risk.cvar_gate import get_cvar_gate
from regime.controller import get_regime_controller

logger = logging.getLogger("ALLOCATOR")

class InstitutionalAllocator:
    """
    Allocates capital using Fractional Kelly and Constraints.
    Integrated with CapitalAuctionEngine for deterministic competition.
    """
    def __init__(self, risk_manager=None, max_position_pct=0.10, gamma=0.5, max_leverage=1.0, hurdle_rate=0.02, **kwargs):
        self.risk_manager = risk_manager
        self.max_pos = max_position_pct
        self.gamma = gamma
        self.max_leverage = max_leverage
        self.auction_engine = CapitalAuctionEngine(hurdle_rate=hurdle_rate, total_cap_limit=max_leverage, gamma=gamma)

        # PM Brain Integration (High-Priority)
        from portfolio.pm_brain import PMBrain
        # FIX 5: Bandwidth Expansion (12-15 positions)
        self.pm_brain = PMBrain(cvar_limit=-0.05, max_positions=15)

        # Institutional Infrastructure Integration
        self.cvar_gate = get_cvar_gate()
        self.regime_controller = get_regime_controller()
        logger.info("[ALLOCATOR] Institutional modules wired: CVaRGate, RegimeController")
        logger.info("[ALLOCATOR] PM Brain initialized for capital competition")

    def allocate(self, *args, **kwargs):
        """
        Unified allocate method supporting:
        1. Single: allocate(request: AllocationRequest) -> Dict
        2. Batch: allocate(requests: List[AllocationRequest]) -> Dict[str, float]
        3. Legacy: allocate(signals, prices, volumes, portfolio, ts, metadata, method) -> OrderList
        """
        if len(args) == 1:
            if isinstance(args[0], AllocationRequest):
                return self._allocate_new(args[0])
            elif isinstance(args[0], list) and all(isinstance(x, AllocationRequest) for x in args[0]):
                return self.allocate_batch(args[0])
            elif isinstance(args[0], pd.DataFrame):
                # Convert DataFrame signals to requests
                requests = self._signals_to_requests(args[0], kwargs.get('data'), kwargs.get('metadata', {}))
                prices = kwargs.get('data') if kwargs.get('data') is not None else kwargs.get('prices')
                return self.allocate_batch(requests, prices=prices)

        return self._allocate_legacy(*args, **kwargs)

    def allocate_batch(self, requests: List[AllocationRequest], prices: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Runs the Capital Auction for multiple competing requests.
        Optionally applies Hedging overlay if RiskManager supports it.
        """
        # FIX 6: Satellite Alpha Sleeve (Controlled Offensive)
        # Split capital: 90% Core, 10% Satellite
        SATELLITE_CAP_PCT = 0.10
        CORE_CAP_PCT = 0.90

        # Identify Satellite Candidates (Momentum/Breakout metadata)
        # If no explicit metadata, use high-conviction heuristic (mu > top decile)
        satellite_reqs = []
        core_reqs = []

        # Sort by signal strength (mu)
        sorted_reqs = sorted(requests, key=lambda x: x.mu, reverse=True)

        # Take top 2 as potential satellite if high conviction, else treat as core
        # This is a simple heuristic since we lack explicit strategy tags on requests yet
        for i, req in enumerate(sorted_reqs):
            is_satellite = False
            if i < 2 and req.confidence > 0.7: # Top 2 high conviction
                 is_satellite = True

            if is_satellite:
                satellite_reqs.append(req)
            else:
                core_reqs.append(req)

        # Auction Core (90% Cap)
        # We simulate this by scaling the resulting weights, or passing a cap to auction engine?
        # Auction engine expects to allocate up to 'total_cap_limit'.
        # We'll allocate fully then scale.

        core_weights = self.auction_engine.auction_capital(core_reqs)
        # Scale core to 90%
        core_weights = {k: v * CORE_CAP_PCT for k, v in core_weights.items()}

        # Satellite Allocation (Simple Equal Weight to survivors)
        satellite_weights = {}
        if satellite_reqs:
            sat_per_share = SATELLITE_CAP_PCT / len(satellite_reqs)
            for req in satellite_reqs:
                satellite_weights[req.symbol] = sat_per_share

        # Combine
        weights = {**core_weights, **satellite_weights}

        # Phase D: Hedging Overlay
        if self.risk_manager and hasattr(self.risk_manager, 'sector_hedger') and weights:
            # 1. Sector Hedge
            sector_hedges = self.risk_manager.sector_hedger.calculate_hedge_overlay(weights)
            for s, w in sector_hedges.items():
                weights[s] = weights.get(s, 0.0) + w

            # 2. Beta Hedge
            if prices is not None:
                beta_hedges = self.risk_manager.beta_neutralizer.calculate_beta_neutralization(weights, prices)
                for s, w in beta_hedges.items():
                    weights[s] = weights.get(s, 0.0) + w

        return weights

    def _allocate_new(self, request: AllocationRequest) -> Dict[str, Any]:
        """
        Single request interface.
        """
        # We can just wrap it in a list and use the auction engine for consistency
        weights = self.allocate_batch([request])
        weight = weights.get(request.symbol, 0.0)

        return {
            "symbol": request.symbol,
            "quantity": weight,
            "order_type": "MARKET",
            "metadata": {
                "mu": request.mu,
                "sigma": request.sigma,
                "confidence": request.confidence
            }
        }

    def _signals_to_requests(self, signals: pd.DataFrame, data: Optional[pd.DataFrame], metadata: Dict) -> List[AllocationRequest]:
        """
        Heuristic: Convert basic signal DataFrame to AllocationRequest objects.
        """
        requests = []
        if signals.empty:
            return []

        idx_val = signals.index[-1]
        if hasattr(idx_val, 'isoformat'):
            timestamp = idx_val.isoformat()
        else:
            try:
                timestamp = pd.to_datetime(idx_val).isoformat()
            except:
                timestamp = str(pd.Timestamp.utcnow())

        # signals is expected to be a DataFrame where columns are tickers
        for symbol in signals.columns:
            try:
                val = float(signals[symbol].iloc[-1])
            except (ValueError, TypeError):
                logger.warning(f"Allocator: Invalid signal value for {symbol}, defaulting to 0.5")
                val = 0.5

            if np.isnan(val) or np.isinf(val):
                logger.warning(f"Allocator: NaN signal for {symbol}, skipping/defaulting")
                val = 0.5

            # Mapping val (0.0 to 1.0) to mu (-0.05 to 0.05) assuming 0.5 is neutral
            mu = (val - 0.5) * 0.1

            # Simple defaults for missing dimensions
            sigma = 0.02 # 2% daily vol default
            confidence = 0.5
            if val > 0.8 or val < 0.2: confidence = 0.8 # Higher confidence on extremes

            requests.append(AllocationRequest(
                symbol=symbol,
                mu=mu,
                sigma=sigma,
                confidence=confidence,
                liquidity=1000000.0,
                regime="NORMAL",
                timestamp=timestamp,
                metadata=metadata.get(symbol, {})
            ))
        return requests

    def _allocate_legacy(self, signals, prices, volumes, portfolio, ts, metadata=None, method="risk_parity"):
        """
        Legacy interface for backward compatibility.
        """
        from dataclasses import dataclass
        @dataclass
        class OrderList:
            orders: List[Any]

            def __len__(self):
                """Enable len() support for OrderList (Sized protocol)."""
                return len(self.orders)

        # Try to use signals as the modern batch if it's a DataFrame
        if isinstance(signals, pd.DataFrame):
            # Convert signals DataFrame to requests and run capital auction
            # This ensures we get Dict[str, float] return type directly
            requests = self._signals_to_requests(signals, prices, metadata or {})
            weights = self.allocate_batch(requests, prices=prices)

            # Simplified order conversion
            orders = []
            for symbol, w in weights.items():
                if abs(w) > 0.0001:
                    orders.append({"symbol": symbol, "quantity": w})
            return OrderList(orders=orders)

        return OrderList(orders=[])


# agents/decision_agent.py
"""
DecisionAgent: production-grade deterministic trading decision agent.

Usage (CLI):
  python -m agents.decision_agent --input sample_input.json --out proposal.json

API:
  from agents.decision_agent import DecisionAgent
  agent = DecisionAgent(agent_id="agent_meanrev_v2")
  proposal = agent.run(input_dict)

Notes:
 - Attempts to call risk.quantum.path_integral and risk.quantum.entanglement_detector if available.
 - Deterministic: seeds derived from run_id + agent_id.
 - Outputs strict JSON proposal schema (see spec).
"""
from __future__ import annotations
import json, hashlib, argparse, math, time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from copy import deepcopy

# Optional imports from earlier modules; guarded
try:
    from risk.quantum.path_integral import gaussian_sample_q, weight_p_over_q_gaussian, importance_sample_paths, weighted_cvar_from_losses
    PATH_INTEGRAL_AVAILABLE = True
except Exception:
    PATH_INTEGRAL_AVAILABLE = False

try:
    from risk.quantum.entanglement_detector import build_entanglement_matrix, entanglement_indices
    ENTANGLEMENT_AVAILABLE = True
except Exception:
    ENTANGLEMENT_AVAILABLE = False

# CONSTANTS
CONTRACT_VERSION = "1.0"
SCHEMA_HASH_SEED = "decision_agent_schema_v1"

def _sha256_hex(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(s.encode()).hexdigest()

def deterministic_rng(run_id: str, agent_id: str, base_seed: int = 0) -> np.random.Generator:
    h = hashlib.sha256(f"{run_id}|{agent_id}|{base_seed}".encode()).hexdigest()
    seed = int(h[:16], 16) % (2**31 - 1)
    return np.random.default_rng(seed)

@dataclass
class DecisionProposal:
    run_id: str
    seed: int
    timestamp: str
    agent_id: str
    decision: str
    confidence: float
    ensemble_score: float
    primary_signal: str
    suggested_notional_pct: float
    suggested_qty: int
    price_limits: Dict[str, float]
    entry_zone: Dict[str, Any]
    exit_logic: Dict[str, Any]
    risk_checks: Dict[str, bool]
    explain: Dict[str, Any]
    warnings: List[str]
    contract_version: str = CONTRACT_VERSION
    schema_hash: str = ""
    signature: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=float)

class DecisionAgent:
    def __init__(self, agent_id: str = "agent_prod_v1", base_seed: int = 0, config: Optional[Dict] = None):
        self.agent_id = agent_id
        self.base_seed = base_seed
        self.config = {
            "ensemble_weights": {"mr":0.45, "mom":0.35, "liq":0.1, "vol":0.1},
            "mr_buy_threshold": 0.65,
            "ensemble_buy_threshold": 0.65,
            "ensemble_sell_threshold": -0.5,
            "min_edge_bps": 10.0,   # minimum net edge after costs
            "bootstrap_samples": 400,
            "bootstrap_block": 20,
            "fractional_kelly_cap": 0.25,
            "per_symbol_max_pct": 0.02,
            "strategy_max_pct": 0.05,
            "use_path_integral": True,
            "path_integral_samples": 400,
            "entanglement_threshold": 0.7,
            "entanglement_beta": 0.5,
            "min_data_confidence": 0.6,
            "safety_buffer_bps": 5.0
        }
        if config:
            self.config.update(config)

    # -----------------------
    # Utilities
    # -----------------------
    def _now_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _derive_seed(self, run_id: str) -> int:
        rng = deterministic_rng(run_id, self.agent_id, self.base_seed)
        return int(rng.integers(1, 2**31 - 1))

    # -----------------------
    # Core math pieces
    # -----------------------
    def _ensemble_components(self, features: Dict[str, float]) -> Dict[str, float]:
        # Mean-reversion component (higher when oversold)
        boll_z = float(features.get("boll_z", 0.0))
        rsi3 = float(features.get("rsi_3", 50.0))
        mr_comp = (-0.3 * boll_z) + (0.5 * max(0.0, (30.0 - rsi3) / 30.0))
        # Momentum component (ema gap + macd)
        ema9 = float(features.get("ema_9", 0.0))
        ema21 = float(features.get("ema_21", 1.0))
        ema_gap_pct = (ema9 / ema21 - 1.0) if ema21 != 0 else 0.0
        macd_hist = float(features.get("macd_hist", 0.0))
        mom_comp = 0.6 * ema_gap_pct + 0.4 * macd_hist
        # Liquidity component
        vol_z = float(features.get("volume_z", 0.0))
        adv_ok = features.get("adv_ok", True)
        liq_comp = 0.3 if (adv_ok and vol_z > 1.0) else -0.2
        # Volatility component (penalize high ATR)
        atr = float(features.get("atr_pct", 0.0))
        vol_comp = -0.5 if atr > 3.5 else 0.1
        # clamp each to [-1,1]
        def clamp(x): return max(-1.0, min(1.0, x))
        comps = {
            "mr": clamp(mr_comp),
            "mom": clamp(mom_comp),
            "liq": clamp(liq_comp),
            "vol": clamp(vol_comp),
            "ema_gap_pct": ema_gap_pct
        }
        return comps

    def _ensemble_score(self, comps: Dict[str, float]) -> float:
        w = self.config["ensemble_weights"]
        score = w["mr"]*comps["mr"] + w["mom"]*comps["mom"] + w["liq"]*comps["liq"] + w["vol"]*comps["vol"]
        # normalize roughly into [-1,1]
        return float(max(-1.0, min(1.0, score)))

    def _bootstrap_mu_sigma(self, price_series: List[float], rng: np.random.Generator, n_samples:int=400, block:int=20) -> Tuple[float,float]:
        # returns mu (mean return) and sigma (std of returns) annualized approx
        # use block bootstrap on log-returns
        if len(price_series) < 2:
            return 0.0, 0.0001
        p = np.array(price_series)
        logret = np.diff(np.log(p))
        T = len(logret)
        if T <= 0:
            return 0.0, 0.0001
        samples = []
        for _ in range(n_samples):
            # block bootstrap
            idxs = rng.integers(0, max(1, T-block+1), size=math.ceil(T/block))
            blocks = [logret[i:i+block] for i in idxs]
            sample = np.concatenate(blocks)[:T]
            samples.append(np.mean(sample))
        mus = np.array(samples)
        mu = float(np.mean(mus))
        sigma = float(np.std(mus)) if np.std(mus) > 0 else 1e-6
        # annualize: assume 252 trading days, daily returns
        mu_a = mu * 252.0
        sigma_a = sigma * math.sqrt(252.0)
        # conservative shrink/inflate
        mu_a = mu_a * 0.5
        sigma_a = sigma_a * 1.5
        return mu_a, sigma_a

    def _fractional_kelly_pct(self, mu: float, sigma: float, cap_frac: float = 0.25) -> float:
        """
        Conservative fractional Kelly for continuous returns approx:
        naive kelly f* = mu / sigma^2  (for normal approx)
        limit to [0, cap_frac]
        """
        if sigma <= 0:
            return 0.0
        naive = mu / (sigma * sigma)
        f = max(0.0, naive * 0.5)  # half-kelly as base
        return min(cap_frac, float(f))

    # -----------------------
    # Stress & entanglement gates
    # -----------------------
    def _path_integral_cvar_estimate(self, hist_returns: np.ndarray, price_mean: np.ndarray, rng: np.random.Generator, samples:int=400) -> Optional[float]:
        if not PATH_INTEGRAL_AVAILABLE or samples <= 0:
            return None
        # simple Gaussian proposal usage via risk.quantum.path_integral helpers
        mu = np.mean(hist_returns, axis=0)
        cov = np.cov(hist_returns.T) + np.eye(hist_returns.shape[1]) * 1e-8
        sampler = gaussian_sample_q(mu, cov, bias_scale=2.0)
        def wfunc(tau): return weight_p_over_q_gaussian(tau, mu, cov)
        samples_tau, weights = importance_sample_paths(sampler, wfunc, samples)
        # map samples to portfolio losses: assume single-asset scenario => use norm of x as loss proxy
        losses = np.array([np.linalg.norm(t["x"]) for t in samples_tau])
        cvar = weighted_cvar_from_losses(losses, weights, alpha=0.95)
        return float(cvar)

    def _entanglement_check(self, hist_prices: np.ndarray) -> Tuple[Optional[float], Optional[List[float]]]:
        if not ENTANGLEMENT_AVAILABLE:
            return None, None
        # hist_prices shape (T,) for single symbol -> entanglement needs multivariate; caller may provide matrix
        # fallback: return low entanglement if not available
        try:
            # expecting 2D returns_matrix (n_assets, T); if single-asset, return zeros
            if hist_prices.ndim == 1:
                return 0.0, [0.0]
            returns_matrix = hist_prices
            E = build_entanglement_matrix(returns_matrix, q=0.05)
            indices, global_score = entanglement_indices(E)
            return float(global_score), [float(x) for x in indices.tolist()]
        except Exception:
            return None, None

    # -----------------------
    # Main run
    # -----------------------
    def run(self, input_obj: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(input_obj.get("run_id", f"run_{int(time.time())}"))
        seed = self._derive_seed(run_id)
        rng = deterministic_rng(run_id, self.agent_id, self.base_seed)
        ts = input_obj.get("timestamp", self._now_iso())
        price = float(input_obj.get("price", 0.0))
        nav = float(input_obj.get("nav_usd", 1_000_000.0))
        symbol = str(input_obj.get("symbol", "UNKNOWN"))
        features = input_obj.get("features", {})
        models = input_obj.get("models", {})
        risk = input_obj.get("risk", {})
        execution = input_obj.get("execution", {})
        historical = input_obj.get("historical", {})
        position_state = input_obj.get("position_state", {})
        data_confidence = float(input_obj.get("data_confidence", 1.0)) if input_obj.get("data_confidence") is not None else 1.0

        warnings = []
        # quick data confidence gate
        data_conf_ok = data_confidence >= self.config["min_data_confidence"]
        if not data_conf_ok:
            decision = "REJECT"
            proposal = DecisionProposal(
                run_id=run_id, seed=seed, timestamp=ts, agent_id=self.agent_id,
                decision=decision, confidence=0.0, ensemble_score=0.0, primary_signal="none",
                suggested_notional_pct=0.0, suggested_qty=0, price_limits={"min":0.0,"max":0.0},
                entry_zone={"low":0.0,"high":0.0,"type":"LIMIT"},
                exit_logic={}, risk_checks={"data_confidence_ok":False,"cvar_ok":False,"entanglement_ok":True,"execution_cost_ok":False},
                explain={"reason":"low_data_confidence","data_confidence":data_confidence}, warnings=["low_data_confidence"]
            )
            proposal.schema_hash = _sha256_hex(SCHEMA_HASH_SEED)
            proposal.signature = _sha256_hex(asdict(proposal))
            return json.loads(proposal.to_json())

        # compute components and ensemble
        comps = self._ensemble_components(features)
        ensemble = self._ensemble_score(comps)

        # primary signal selection
        primary_signal = "mean_reversion" if abs(comps["mr"]) >= abs(comps["mom"]) else "momentum"

        # Estimate distributional mu/sigma from historical price series
        price_series = historical.get("price_series", []) or []
        mu_a, sigma_a = self._bootstrap_mu_sigma(price_series, rng, n_samples=self.config["bootstrap_samples"], block=self.config["bootstrap_block"])

        # expected edge in decimal and bps
        expected_edge_pct = mu_a  # annualized
        expected_edge_bps = expected_edge_pct * 10000.0

        # execution costs
        slippage_bps = float(execution.get("slippage_bps", 6.0))
        spread_bps = float(execution.get("spread_bps", 3.0))
        total_cost_bps = slippage_bps + spread_bps + self.config["safety_buffer_bps"]

        # entanglement check (if matrix provided in historical returns)
        ent_score, ent_indices = None, None
        if ENTANGLEMENT_AVAILABLE and isinstance(historical.get("returns_matrix", None), list):
            hist_mat = np.array(historical["returns_matrix"])
            ent_score, ent_indices = self._entanglement_check(hist_mat)

        ent_ok = True
        tightened_cap_pct = self.config["per_symbol_max_pct"]
        if ent_score is not None:
            if ent_score > self.config["entanglement_threshold"]:
                ent_ok = False
                tightened_cap_pct = max(0.0, tightened_cap_pct * (1 - self.config["entanglement_beta"] * (ent_indices[0] if ent_indices else 1.0)))
                warnings.append(f"entanglement_high:{ent_score:.3f}")

        # path-integral stress CVaR estimate (optional)
        cvar_est = None
        if self.config["use_path_integral"] and PATH_INTEGRAL_AVAILABLE and historical.get("returns_matrix", None) is not None:
            try:
                hist = np.array(historical["returns_matrix"])
                cvar_est = self._path_integral_cvar_estimate(hist, np.array([price]), rng, samples=self.config["path_integral_samples"])
            except Exception as e:
                cvar_est = None
                warnings.append("path_integral_failed")

        # decide buy/sell/hold/reject
        decision = "HOLD"
        confidence = 0.0
        suggested_pct = 0.0
        suggested_qty = 0
        price_min = max(0.0, price * 0.997)
        price_max = price * 1.003

        # BUY logic
        if ensemble >= self.config["ensemble_buy_threshold"] and comps["mr"] >= self.config["mr_buy_threshold"]:
            # check edge vs costs
            if expected_edge_bps * data_confidence > (total_cost_bps + self.config["min_edge_bps"]):
                # compute fractional Kelly
                fkelly = self._fractional_kelly_pct(mu_a, sigma_a, cap_frac=self.config["fractional_kelly_cap"])
                # volatility target: use ATR_pct if present
                atr_pct = float(features.get("atr_pct", 1.5))
                target_vol_pct = 0.15  # strategy level target annual vol
                # predicted optimal pct by volatility (very conservative)
                vol_based_pct = min(0.5, (target_vol_pct / max(0.0001, atr_pct/100.0)))  # rough
                suggested_pct = min(tightened_cap_pct, vol_based_pct * fkelly)
                # hard caps
                suggested_pct = min(suggested_pct, self.config["per_symbol_max_pct"], self.config["strategy_max_pct"])
                # if extremely small, reject
                if suggested_pct * nav < 100.0:
                    decision = "REJECT"
                    confidence = 0.0
                    warnings.append("notional_too_small_after_sizing")
                else:
                    decision = "BUY"
                    confidence = float(min(0.99, abs(ensemble)))
                    suggested_qty = int(math.floor((nav * suggested_pct) / price))
            else:
                decision = "REJECT"
                confidence = 0.0
                warnings.append("insufficient_net_edge_after_costs")

        # SELL logic: if has position and sell signals or stop triggered (caller should provide position state)
        elif position_state.get("has_position", False):
            # if ensemble strongly negative or price beat target or stop-level reached
            if ensemble <= self.config["ensemble_sell_threshold"]:
                decision = "SELL"
                confidence = min(0.99, abs(ensemble))
                suggested_pct = position_state.get("position_pct", 0.0) or 0.0
                suggested_qty = int(position_state.get("qty", 0))
            # if unrealized negative beyond stop_distance_pct
            u_pct = position_state.get("unrealized_pct")
            stop_dist = float(input_obj.get("risk", {}).get("stop_distance_pct", 0.02))
            if u_pct is not None and u_pct <= -abs(stop_dist):
                decision = "SELL"
                confidence = 0.95
                suggested_qty = int(position_state.get("qty", 0))

        # final risk gate: cvar check (if available)
        cvar_ok = True
        if cvar_est is not None:
            # simple gate: single-trade implied CVaR contribution must be small fraction of global CVaR limit
            global_cvar_limit = float(risk.get("cvar_limit", 0.05))
            # normalize cvar_est (note: path-integral returns loss proxy not normalized; use conservative test)
            if cvar_est > global_cvar_limit * 10:  # arbitrary protective scaling
                cvar_ok = False
                warnings.append("path_integral_cvar_exceeds_limit")
        # execution cost check
        exec_ok = (expected_edge_bps * data_confidence) > (total_cost_bps + self.config["min_edge_bps"])
        if not exec_ok:
            if decision == "BUY":
                decision = "REJECT"
                warnings.append("execution_costs_too_high")

        # assemble exit logic
        stop_loss_pct = float(input_obj.get("risk", {}).get("stop_distance_pct", 0.02))
        tp1 = 0.03
        tp2 = 0.08
        exit_logic = {
            "stop_loss_price": round(price * (1 - stop_loss_pct), 5),
            "stop_loss_pct": stop_loss_pct,
            "take_profit_tiers": [{"pct":tp1, "qty_frac":0.5}, {"pct":tp2, "qty_frac":0.5}],
            "trailing_pct": max(0.005, 1.2 * float(features.get("atr_pct", 1.5))/100.0),
            "time_in_trade_limit_minutes": int(input_obj.get("risk", {}).get("time_in_trade_limit_minutes", 1440))
        }

        risk_checks = {"data_confidence_ok": True, "cvar_ok": cvar_ok, "entanglement_ok": ent_ok, "execution_cost_ok": exec_ok}

        explain = {
            "why_price_is_mispriced": f"Ensemble={ensemble:.3f}; MR_comp={comps['mr']:.3f}; MOM_comp={comps['mom']:.3f}",
            "indicators_used": {"rsi_3": features.get("rsi_3"), "boll_z": features.get("boll_z"), "ema_gap_pct": comps.get("ema_gap_pct"), "atr_pct": features.get("atr_pct")},
            "expected_edge_bps": expected_edge_bps,
            "expected_return_pct": mu_a,
            "model_versions": {"ensemble": "ens_v1", "bootstrap": f"bs{self.config['bootstrap_samples']}"}
        }

        # create proposal dataclass
        proposal = DecisionProposal(
            run_id=run_id, seed=seed, timestamp=ts, agent_id=self.agent_id,
            decision=decision, confidence=float(confidence), ensemble_score=float(ensemble),
            primary_signal=primary_signal, suggested_notional_pct=float(suggested_pct),
            suggested_qty=int(suggested_qty), price_limits={"min":price_min,"max":price_max},
            entry_zone={"low":price_min,"high":price_max,"type":"LIMIT"},
            exit_logic=exit_logic, risk_checks=risk_checks,
            explain=explain, warnings=warnings
        )
        # schema hash & signature
        proposal.schema_hash = _sha256_hex(SCHEMA_HASH_SEED)
        # signature: sign the JSON payload hash (placeholder deterministic hash)
        sig_payload = {
            "run_id": proposal.run_id, "agent_id": proposal.agent_id, "decision": proposal.decision,
            "seed": proposal.seed, "timestamp": proposal.timestamp
        }
        proposal.signature = _sha256_hex(sig_payload)
        return json.loads(proposal.to_json())

# CLI support
def _load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def _write_json_file(path: str, obj: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input JSON file with schema")
    parser.add_argument("--out", required=True, help="output proposal JSON")
    parser.add_argument("--agent_id", default="agent_prod_v1")
    args = parser.parse_args()
    inp = _load_json_file(args.input)
    agent = DecisionAgent(agent_id=args.agent_id)
    proposal = agent.run(inp)
    _write_json_file(args.out, proposal)
    print("Wrote proposal to", args.out)

if __name__ == "__main__":
    main_cli()

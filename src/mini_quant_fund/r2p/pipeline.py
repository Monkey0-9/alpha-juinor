import logging
import json
import os
from enum import Enum
from typing import Any, Dict, List

logger = logging.getLogger("R2P_PIPELINE")

class ValidationStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"

class R2PResult:
    def __init__(self, model_id: str, status: ValidationStatus, gates: Dict[str, bool], message: str):
        self.model_id = model_id
        self.status = status
        self.gates = gates
        self.message = message

class PromotionPipeline:
    """
    The Gatekeeper.
    Enforces Phases 0-4 requirements before a model can enter Paper or Live trading.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Default Gates
        self.gates = {
            "data_quality": True,
            "reproducibility": True,
            "performance": True,
            "tail_risk": True,
            "governance_signoff": False  # Specific models might skip if pre-approved
        }

    def run_promotion_checks(self, model_urn: str, artifacts_path: str) -> R2PResult:
        """
        Run all gates on a candidate model.
        """
        logger.info(f"[R2P] Starting promotion checks for {model_urn}")
        results = {}

        # 1. Data Quality Gate
        results["data_quality"] = self._check_data_quality(artifacts_path)

        # 2. Reproducibility Gate
        results["reproducibility"] = self._check_reproducibility(model_urn)

        # 3. Performance Gate
        results["performance"] = self._check_performance(artifacts_path)

        # 4. Tail Risk Gate
        for_tail_risk = self._check_tail_risk(artifacts_path)
        results["tail_risk"] = for_tail_risk

        # Aggregation
        all_passed = all(results.values())
        status = ValidationStatus.PASS if all_passed else ValidationStatus.FAIL

        msg = f"Model {model_urn} {'PASSED' if all_passed else 'FAILED'} R2P gates."
        logger.info(msg)

        return R2PResult(model_urn, status, results, msg)

    def _check_data_quality(self, artifacts_path: str) -> bool:
        """Verify training data met DQ standards."""
        try:
            # Load data quality metrics from artifacts
            dq_file = os.path.join(artifacts_path, "data_quality.json")
            if not os.path.exists(dq_file):
                logger.warning(f"No data_quality.json found at {artifacts_path}")
                return False

            with open(dq_file, 'r') as f:
                dq_metrics = json.load(f)

            # Check minimum data quality thresholds
            min_completeness = self.config.get('min_data_completeness', 0.95)
            max_missing_ratio = self.config.get('max_missing_ratio', 0.05)
            min_coverage_days = self.config.get('min_coverage_days', 252)

            completeness = dq_metrics.get('completeness', 0)
            missing_ratio = dq_metrics.get('missing_ratio', 1.0)
            coverage_days = dq_metrics.get('coverage_days', 0)

            if completeness < min_completeness:
                logger.error(f"Data completeness {completeness:.3f} < {min_completeness:.3f}")
                return False

            if missing_ratio > max_missing_ratio:
                logger.error(f"Missing ratio {missing_ratio:.3f} > {max_missing_ratio:.3f}")
                return False

            if coverage_days < min_coverage_days:
                logger.error(f"Coverage days {coverage_days} < {min_coverage_days}")
                return False

            logger.info(f"Data quality check passed: completeness={completeness:.3f}, coverage={coverage_days} days")
            return True

        except Exception as e:
            logger.error(f"Data quality check failed: {e}")
            return False

    def _check_reproducibility(self, model_urn: str) -> bool:
        """Verify the model has a seed and config hash."""
        try:
            # Load model metadata
            metadata_file = f"models/{model_urn}/metadata.json"
            if not os.path.exists(metadata_file):
                logger.error(f"No metadata.json found for model {model_urn}")
                return False

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Check required reproducibility fields
            required_fields = ['random_seed', 'config_hash', 'model_hash', 'training_timestamp']
            missing_fields = [field for field in required_fields if field not in metadata]

            if missing_fields:
                logger.error(f"Missing reproducibility fields: {missing_fields}")
                return False

            # Verify seed is set
            seed = metadata['random_seed']
            if seed is None or seed == '':
                logger.error("Random seed is not set")
                return False

            # Verify config hash exists (non-empty)
            config_hash = metadata['config_hash']
            if not config_hash or len(config_hash) < 8:
                logger.error("Invalid config hash")
                return False

            # Verify model hash exists
            model_hash = metadata['model_hash']
            if not model_hash or len(model_hash) < 8:
                logger.error("Invalid model hash")
                return False

            logger.info(f"Reproducibility check passed for {model_urn}: seed={seed}, config_hash={config_hash[:8]}...")
            return True

        except Exception as e:
            logger.error(f"Reproducibility check failed: {e}")
            return False

    def _check_performance(self, artifacts_path: str) -> bool:
        """Verify OOS metrics exceed baseline."""
        try:
            # Load performance metrics from artifacts
            metrics_file = os.path.join(artifacts_path, "metrics.json")
            if not os.path.exists(metrics_file):
                logger.error(f"No metrics.json found at {artifacts_path}")
                return False

            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Get baseline performance thresholds from mini_quant_fund.config
            min_sharpe = self.config.get('min_sharpe_ratio', 0.5)
            min_sortino = self.config.get('min_sortino_ratio', 0.8)
            max_max_drawdown = self.config.get('max_max_drawdown', 0.25)
            min_win_rate = self.config.get('min_win_rate', 0.52)

            # Extract OOS metrics
            oos_sharpe = metrics.get('oos_sharpe_ratio', 0)
            oos_sortino = metrics.get('oos_sortino_ratio', 0)
            oos_max_drawdown = metrics.get('oos_max_drawdown', 1.0)
            oos_win_rate = metrics.get('oos_win_rate', 0)

            # Check each metric against baseline
            if oos_sharpe < min_sharpe:
                logger.error(f"OOS Sharpe {oos_sharpe:.3f} < baseline {min_sharpe:.3f}")
                return False

            if oos_sortino < min_sortino:
                logger.error(f"OOS Sortino {oos_sortino:.3f} < baseline {min_sortino:.3f}")
                return False

            if oos_max_drawdown > max_max_drawdown:
                logger.error(f"OOS Max Drawdown {oos_max_drawdown:.3f} > baseline {max_max_drawdown:.3f}")
                return False

            if oos_win_rate < min_win_rate:
                logger.error(f"OOS Win Rate {oos_win_rate:.3f} < baseline {min_win_rate:.3f}")
                return False

            logger.info(f"Performance check passed: Sharpe={oos_sharpe:.3f}, Sortino={oos_sortino:.3f}, MaxDD={oos_max_drawdown:.3f}")
            return True

        except Exception as e:
            logger.error(f"Performance check failed: {e}")
            return False

    def _check_tail_risk(self, artifacts_path: str) -> bool:
        """Verify CVaR / Drawdown is within safety limits."""
        try:
            # Load backtest metrics from artifacts
            backtest_file = os.path.join(artifacts_path, "backtest_metrics.json")
            if not os.path.exists(backtest_file):
                logger.error(f"No backtest_metrics.json found at {artifacts_path}")
                return False

            with open(backtest_file, 'r') as f:
                backtest_metrics = json.load(f)

            # Get tail risk thresholds from mini_quant_fund.config
            max_cvar_95 = self.config.get('max_cvar_95', 0.05)  # 5% max CVaR at 95%
            max_cvar_99 = self.config.get('max_cvar_99', 0.10)  # 10% max CVaR at 99%
            max_drawdown = self.config.get('max_tail_drawdown', 0.20)  # 20% max drawdown
            min_var_days = self.config.get('min_var_coverage_days', 252)  # 1 year minimum

            # Extract tail risk metrics
            cvar_95 = backtest_metrics.get('cvar_95', 1.0)
            cvar_99 = backtest_metrics.get('cvar_99', 1.0)
            max_dd = backtest_metrics.get('max_drawdown', 1.0)
            var_coverage_days = backtest_metrics.get('var_coverage_days', 0)

            # Check CVaR thresholds
            if cvar_95 > max_cvar_95:
                logger.error(f"CVaR 95% {cvar_95:.3f} exceeds threshold {max_cvar_95:.3f}")
                return False

            if cvar_99 > max_cvar_99:
                logger.error(f"CVaR 99% {cvar_99:.3f} exceeds threshold {max_cvar_99:.3f}")
                return False

            # Check drawdown
            if max_dd > max_drawdown:
                logger.error(f"Max drawdown {max_dd:.3f} exceeds threshold {max_drawdown:.3f}")
                return False

            # Check VaR coverage period
            if var_coverage_days < min_var_days:
                logger.error(f"VaR coverage {var_coverage_days} days below minimum {min_var_days}")
                return False

            logger.info(f"Tail risk check passed: CVaR95={cvar_95:.3f}, CVaR99={cvar_99:.3f}, MaxDD={max_dd:.3f}")
            return True

        except Exception as e:
            logger.error(f"Tail risk check failed: {e}")
            return False

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()

class QualityAgent:
    """
    Ruthless Data Quality Auditor.
    Uses institutional priors (Reconstruction Error / Isolation Forest)
    to compute quality_score = exp(-recon_error).
    """
    def __init__(self, recon_error_threshold: float = 0.5):
        self.threshold = recon_error_threshold

    def compute_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculates data_quality_score in [0, 1].
        If score < 0.6, triggers 75% reduction in sizing.
        """
        if df is None or (hasattr(df, "empty") and df.empty):
            return 0.0

        # Institutional Checks
        errors = []

        # 1. Integrity check (No-NaNs, No-Zeros)
        null_ratio = float(df.isnull().values.mean())
        # Use .values to get a flat numpy array for truthiness
        zero_prices = int((df["Close"] <= 0).values.sum())

        # 2. Statistical Anomaly Check (Isolation Forest Proxy)
        # Using rolling 6-sigma volume spikes as reconstruction error proxy
        median_vol = df["Volume"].rolling(window=180).median()
        std_vol = df["Volume"].rolling(window=180).std() + 1e-9
        z_vols = (df["Volume"] - median_vol) / std_vol
        recon_error = float(np.abs(z_vols.tail(5).mean()) / 6.0) # Normalized reconstruction error

        # 3. Quality Formula: score = exp(-recon_error) - penalties
        score = np.exp(-recon_error)

        # Penalties for hard failures
        if zero_prices > 0: score -= 0.4
        if null_ratio > 0.05: score -= 0.3

        # Ensure range [0, 1]
        final_score = float(np.clip(score, 0.0, 1.0))

        logger.info("Data quality computed",
                    score=final_score,
                    recon_error=recon_error,
                    zero_prices=zero_prices)

        return final_score

    def get_validation_report(self, df: pd.DataFrame) -> List[str]:
        reports = []
        if (df["Close"] <= 0).values.any(): reports.append("ZERO_PRICES")
        if df.isnull().values.any(): reports.append("MISSING_DATA")
        return reports

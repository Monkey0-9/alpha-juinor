import unittest
import os
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from mini_quant_fund.database.adapters.sqlite_adapter import SQLiteAdapter
from mini_quant_fund.database.schema import DailyPriceRecord, DecisionRecord
from mini_quant_fund.r2p.pipeline import PromotionPipeline, ValidationStatus

class TestP1Correctness(unittest.TestCase):
    def setUp(self):
        self.db_path = "runtime/test_p1.db"
        self.audit_path = "runtime/test_p1_audit.jsonl"
        self.adapter = SQLiteAdapter(self.db_path, self.audit_path)
        
    def tearDown(self):
        self.adapter.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.audit_path):
            os.remove(self.audit_path)

    def test_batch_upsert_duplicates(self):
        """P1: Verify batch UPSERT prevents tick corruption (duplicates)."""
        records = [
            DailyPriceRecord(symbol="AAPL", date="2025-01-01", open=150.0, high=155.0, low=149.0, close=152.0, adjusted_close=152.0, volume=1000000, vwap=151.0, trade_count=5000, provider="YF", raw_hash="h1", ingestion_timestamp="2025-01-01T00:00:00Z"),
            DailyPriceRecord(symbol="AAPL", date="2025-01-01", open=150.0, high=155.0, low=149.0, close=160.0, adjusted_close=160.0, volume=2000000, vwap=155.0, trade_count=6000, provider="YF", raw_hash="h2", ingestion_timestamp="2025-01-01T01:00:00Z"),
        ]
        
        # Insert batch with duplicate (same symbol/date)
        # The adapter should use ON CONFLICT to update rather than duplicate
        self.adapter.upsert_daily_prices_batch(records)
        
        # Verify only ONE row exists for AAPL 2025-01-01
        df = self.adapter.get_daily_prices("AAPL")
        self.assertEqual(len(df), 1, "Duplicate symbol/date found in price_history")
        self.assertEqual(df.iloc[0]['close'], 160.0, "UPSERT failed to update existing record")

    def test_r2p_pipeline_logic(self):
        """P1: Verify R2P pipeline gates are functional (not stubs)."""
        pipeline = PromotionPipeline(config={"min_oos_sharpe": 1.0})
        
        # Create mock artifacts
        os.makedirs("runtime/test_artifacts", exist_ok=True)
        os.makedirs("models/test_model_v1", exist_ok=True)
        
        with open("models/test_model_v1/metadata.json", "w") as f:
            json.dump({
                "random_seed": 42,
                "config_hash": "hash123456",
                "model_hash": "modelhash123",
                "training_timestamp": "2025-01-01T00:00:00Z"
            }, f)
            
        with open("runtime/test_artifacts/metrics.json", "w") as f:
            json.dump({
                "oos_sharpe_ratio": 1.5, 
                "oos_sortino_ratio": 2.0, 
                "oos_max_drawdown": 0.1, 
                "oos_win_rate": 0.55
            }, f)
            
        with open("runtime/test_artifacts/data_quality.json", "w") as f:
            json.dump({
                "completeness": 0.99, 
                "missing_ratio": 0.01, 
                "coverage_days": 500
            }, f)
            
        with open("runtime/test_artifacts/backtest_metrics.json", "w") as f:
            json.dump({
                "cvar_95": 0.02,
                "cvar_99": 0.05,
                "max_drawdown": 0.1,
                "var_coverage_days": 300
            }, f)
        
        result = pipeline.run_promotion_checks("test_model_v1", "runtime/test_artifacts")
        
        self.assertEqual(result.status, ValidationStatus.PASS)
        
        # Test FAILURE case
        with open("runtime/test_artifacts/metrics.json", "w") as f:
            json.dump({"oos_sharpe_ratio": 0.5}, f) # Below 1.0 threshold
            
        result_fail = pipeline.run_promotion_checks("test_model_v1", "runtime/test_artifacts")
        self.assertEqual(result_fail.status, ValidationStatus.FAIL)

    def test_skipping_history_persistence(self):
        """P1: Verify skipping_history persists in DB and is correctly loaded."""
        # 1. Insert 2 consecutive SKIPS for AAPL
        decisions = [
            DecisionRecord(cycle_id="c1", symbol="AAPL", final_decision="SKIP_TOO_SMALL", reason_codes=["SMALL"], mu_hat=0.01, sigma_hat=0.02, conviction=0.5, position_size=0.0, data_quality_score=1.0, provider_confidence=0.5),
            DecisionRecord(cycle_id="c2", symbol="AAPL", final_decision="SKIP_LOW_CONFIDENCE", reason_codes=["LOW"], mu_hat=0.01, sigma_hat=0.02, conviction=0.2, position_size=0.0, data_quality_score=1.0, provider_confidence=0.5),
        ]
        self.adapter.insert_decisions(decisions)
        
        # 2. Get consecutive skips
        skips = self.adapter.get_consecutive_skips()
        self.assertEqual(skips.get("AAPL"), 2, "Consecutive skips count incorrect or not persisted")
        
        # 3. Insert EXECUTE
        exec_decision = [
            DecisionRecord(cycle_id="c3", symbol="AAPL", final_decision="EXECUTE", reason_codes=["OK"], mu_hat=0.05, sigma_hat=0.02, conviction=0.8, position_size=0.05, data_quality_score=1.0, provider_confidence=0.5),
        ]
        self.adapter.insert_decisions(exec_decision)
        
        # 4. Count should RESET to 0
        skips_reset = self.adapter.get_consecutive_skips()
        self.assertEqual(skips_reset.get("AAPL"), 0, "Consecutive skips should reset on EXECUTE")

if __name__ == "__main__":
    unittest.main()

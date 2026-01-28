
import pytest
import numpy as np
from governance.capital_auction import CapitalAuctionEngine, CapitalAuctionInput
from governance.institutional_specification import AssetClass

class TestCapitalAuctionNaNs:

    @pytest.fixture
    def engine(self):
        return CapitalAuctionEngine()

    def test_nan_mu_rejection(self, engine):
        """Test that NaN expected return leads to rejection"""
        candidate = CapitalAuctionInput(
            symbol="NAN_MU_SYMBOL",
            mu=float('nan'),
            sigma=0.02,
            cvar_95=0.03,
            data_quality_score=0.9,
            history_days=1300,
            strategy_id="test_strat",
            asset_class=AssetClass.STOCKS.value
        )
        candidates = [candidate]
        outputs = engine.run_auction(candidates, portfolio_nav=100000)

        output = outputs["NAN_MU_SYMBOL"]
        assert output.allocated == False
        # Should be rejected
        # We need to verify reason. If not handled, it might crash or produce NaN.
        # Current implementation probably doesn't check for NaN explicitly.

    def test_nan_sigma_rejection(self, engine):
        """Test that NaN sigma leads to rejection"""
        candidate = CapitalAuctionInput(
            symbol="NAN_SIGMA_SYMBOL",
            mu=0.05,
            sigma=float('nan'),
            cvar_95=0.03,
            data_quality_score=0.9,
            history_days=1300,
            strategy_id="test_strat",
            asset_class=AssetClass.STOCKS.value
        )
        candidates = [candidate]
        outputs = engine.run_auction(candidates, portfolio_nav=100000)

        output = outputs["NAN_SIGMA_SYMBOL"]
        assert output.allocated == False

    def test_low_confidence_rejection(self, engine):
        """Test that low confidence leads to rejection (if confidence implemented)"""
        # CapitalAuctionInput doesn't have 'confidence' field directly mapping to simple float,
        # but data_quality_score is used.
        candidate = CapitalAuctionInput(
            symbol="LOW_QUAL_SYMBOL",
            mu=0.05,
            sigma=0.02,
            cvar_95=0.03,
            data_quality_score=0.4, # < 0.6 threshold
            history_days=1300,
            strategy_id="test_strat",
            asset_class=AssetClass.STOCKS.value
        )
        candidates = [candidate]
        outputs = engine.run_auction(candidates, portfolio_nav=100000)

        output = outputs["LOW_QUAL_SYMBOL"]
        assert output.allocated == False

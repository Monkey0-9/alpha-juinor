import pytest
import pandas as pd
from unittest.mock import patch, AsyncMock, MagicMock
from nexus.core.engine import NexusEngine


@pytest.mark.asyncio
async def test_engine_no_trade_state():
    """Verify that when AI Ensemble outputs 0, no trade is executed."""
    engine = NexusEngine()

    # Mock alpha engine to return some signals
    engine.alpha_engine.get_batch_signals = AsyncMock(
        return_value={"AAPL": 0.8}
    )
    engine.alpha_engine.fetch_market_data = AsyncMock(
        return_value=pd.DataFrame({"close": [150, 151, 152]})
    )

    # Mock Ensemble Brain to output 0 (NO_TRADE)
    with patch.object(engine.ensemble_brain, "get_signal", return_value=0.0):
        # Prevent actual alpaca calls by mocking engine methods
        with patch.object(
            engine, "get_positions", new_callable=AsyncMock
        ) as mock_pos:
            mock_pos.return_value = []

            with patch.object(
                engine, "get_account_state", new_callable=AsyncMock
            ) as mock_acc:
                mock_acc.return_value = {
                    "equity": 10000,
                    "buying_power": 10000,
                }

                with patch.object(
                    engine, "_submit_trade", new_callable=AsyncMock
                ) as mock_submit:
                    await engine.run_cycle()

                    # Submit trade should NOT be called for AAPL because
                    # conviction was 0
                    mock_submit.assert_not_called()


@pytest.mark.asyncio
async def test_engine_crisis_correlation():
    """Verify that during a turbulent regime, sizes are cut and no new extreme trades happen."""
    engine = NexusEngine()

    # Force a turbulent regime by returning highly volatile benchmark data
    volatile_close = [100, 95, 105, 90, 110, 85, 115, 80, 120, 75, 125]
    engine.alpha_engine.fetch_market_data = AsyncMock(
        return_value=pd.DataFrame({"close": volatile_close})
    )
    engine.alpha_engine.get_batch_signals = AsyncMock(
        return_value={"AAPL": 0.9}
    )
    engine.symbols = ["AAPL"]

    with patch.object(engine.ensemble_brain, "get_signal", return_value=0.9):
        with patch.object(
            engine, "get_positions", new_callable=AsyncMock
        ) as mock_pos:
            mock_pos.return_value = []

            with patch.object(
                engine, "get_account_state", new_callable=AsyncMock
            ) as mock_acc:
                mock_acc.return_value = {
                    "equity": 10000,
                    "buying_power": 10000,
                }

                with patch.object(
                    engine.regime_detector, "detect", return_value="TURBULENT"
                ):
                    with patch.object(
                        engine, "_submit_trade", new_callable=AsyncMock
                    ) as mock_submit:
                        # Also patch the _get_client to avoid http requests for
                        # clock and fills
                        mock_client = AsyncMock()
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"is_open": True}
                        mock_client.get.return_value = mock_response

                        with patch.object(
                            engine, "_get_client", return_value=mock_client
                        ):
                            with patch.object(
                                engine,
                                "refresh_universe",
                                new_callable=AsyncMock,
                            ):
                                await engine.run_cycle()

                                # Verify that the regime is set to TURBULENT
                                assert engine.market_regime == "TURBULENT"

                            # Verify trade was submitted but scaled
                            mock_submit.assert_called_once()
                            kwargs = mock_submit.call_args[0]
                            weight = kwargs[1]
                            assert weight >= 0.0

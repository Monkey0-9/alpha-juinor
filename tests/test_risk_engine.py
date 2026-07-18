import asyncio
import numpy as np
from nexus.core.engine import NexusEngine
from nexus.math.risk import RiskEngine


def test_risk_engine_assess_risk():
    engine = RiskEngine(confidence_level=0.95)
    returns = np.random.normal(0.001, 0.02, 1000)
    metrics = engine.assess_risk(returns)

    assert isinstance(metrics, dict)
    assert "var" in metrics
    assert metrics["var"] <= 0
    assert metrics["cvar"] <= metrics["var"] + 1e-6
    assert metrics["volatility"] >= 0
    assert isinstance(metrics["sharpe"], float)

def test_manage_positions_threshold():
    engine = NexusEngine(backend_url="http://127.0.0.1:8001")

    async def fake_get_positions():
        return [{"symbol": "AAPL", "unrealized_plpc": 0.09}]

    class FakeClient:
        def __init__(self):
            self.closed_positions = []

        async def delete(self, path):
            self.closed_positions.append(path)
            return {"success": True}

    fake_client = FakeClient()

    async def fake_get_client():
        return fake_client

    engine.get_positions = fake_get_positions
    engine._get_client = fake_get_client

    asyncio.run(engine.manage_positions())
    assert fake_client.closed_positions == ["http://127.0.0.1:8001/api/alpaca/positions/AAPL"]

import pytest

from run_system import MiniQuantFundSystem


@pytest.mark.asyncio
async def test_run_raises_and_still_calls_shutdown(monkeypatch):
    system = MiniQuantFundSystem()
    events = []

    async def initialize_ok():
        events.append("initialize")

    async def health_check_fail():
        events.append("health_check")
        raise RuntimeError("health-check-boom")

    async def shutdown_ok():
        events.append("shutdown")
        system.running = False

    monkeypatch.setattr(system, "initialize", initialize_ok)
    monkeypatch.setattr(system, "run_system_health_check", health_check_fail)
    monkeypatch.setattr(system, "shutdown", shutdown_ok)

    with pytest.raises(RuntimeError, match="health-check-boom"):
        await system.run()

    assert events == ["initialize", "health_check", "shutdown"]

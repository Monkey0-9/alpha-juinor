from dataclasses import dataclass

from mini_quant_fund.performance.fast_signal_selector import select_best_signal_index


@dataclass
class _DummySignal:
    expected_return: float
    conviction: float
    urgency: float
    holding_period: str


def test_select_best_signal_index_prefers_higher_score():
    signals = [
        _DummySignal(expected_return=0.01, conviction=0.9, urgency=0.2, holding_period="days"),
        _DummySignal(expected_return=0.02, conviction=0.7, urgency=0.8, holding_period="hours"),
        _DummySignal(expected_return=0.005, conviction=1.0, urgency=1.0, holding_period="microseconds"),
    ]
    idx = select_best_signal_index(signals)
    assert idx == 2


def test_select_best_signal_index_handles_empty():
    assert select_best_signal_index([]) is None

"""
Optional C++ accelerated signal ranking.

If the shared library is unavailable, this module falls back to pure Python.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any, Iterable, Optional


_HOLDING_PERIOD_CODES = {
    "microseconds": 0,
    "seconds": 1,
    "minutes": 2,
    "hours": 3,
    "days": 4,
    "months": 5,
}


def _score_signal_python(
    expected_return: float,
    conviction: float,
    urgency: float,
    holding_period: str,
) -> float:
    holding_mult = {
        "microseconds": 2.0,
        "seconds": 1.5,
        "minutes": 1.0,
        "hours": 0.7,
        "days": 0.5,
        "months": 0.3,
    }.get(holding_period, 0.5)
    return (expected_return * conviction + urgency * 0.5) * holding_mult


class _CppFastCore:
    def __init__(self) -> None:
        self.lib = self._load()
        self.available = self.lib is not None
        if self.available:
            self.lib.mqf_select_best_signal.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
            ]
            self.lib.mqf_select_best_signal.restype = ctypes.c_int

    @staticmethod
    def _candidate_paths() -> list[Path]:
        env = os.getenv("MQF_CPP_CORE_LIB", "").strip()
        candidates: list[Path] = []
        if env:
            candidates.append(Path(env))

        repo_root = Path(__file__).resolve().parents[3]
        build_dir = repo_root / "cpp" / "fast_decision" / "build"
        candidates.extend(
            [
                build_dir / "fast_decision_core.dll",
                build_dir / "Release" / "fast_decision_core.dll",
                build_dir / "libfast_decision_core.so",
                build_dir / "libfast_decision_core.dylib",
            ]
        )
        return candidates

    def _load(self) -> Optional[ctypes.CDLL]:
        for path in self._candidate_paths():
            if path.exists():
                try:
                    return ctypes.CDLL(str(path))
                except OSError:
                    continue
        return None

    def select_best_index(
        self,
        expected_returns: list[float],
        convictions: list[float],
        urgencies: list[float],
        holding_codes: list[int],
    ) -> Optional[int]:
        if not self.available:
            return None
        n = len(expected_returns)
        if n == 0:
            return None

        expected_arr = (ctypes.c_double * n)(*expected_returns)
        conv_arr = (ctypes.c_double * n)(*convictions)
        urg_arr = (ctypes.c_double * n)(*urgencies)
        hold_arr = (ctypes.c_int * n)(*holding_codes)
        idx = self.lib.mqf_select_best_signal(
            expected_arr,
            conv_arr,
            urg_arr,
            hold_arr,
            n,
        )
        if idx < 0:
            return None
        return int(idx)


_FAST_CORE = _CppFastCore()


def select_best_signal_index(signals: Iterable[Any]) -> Optional[int]:
    """
    Return index of best signal based on expected_return, conviction, urgency and holding_period.
    """
    signal_list = list(signals)
    if not signal_list:
        return None

    expected_returns = [float(getattr(s, "expected_return", 0.0)) for s in signal_list]
    convictions = [float(getattr(s, "conviction", 0.0)) for s in signal_list]
    urgencies = [float(getattr(s, "urgency", 0.0)) for s in signal_list]
    holding_codes = [
        _HOLDING_PERIOD_CODES.get(str(getattr(s, "holding_period", "days")), 4)
        for s in signal_list
    ]

    cpp_idx = _FAST_CORE.select_best_index(
        expected_returns=expected_returns,
        convictions=convictions,
        urgencies=urgencies,
        holding_codes=holding_codes,
    )
    if cpp_idx is not None:
        return cpp_idx

    best_idx = None
    best_score = None
    for i, signal in enumerate(signal_list):
        score = _score_signal_python(
            expected_return=float(getattr(signal, "expected_return", 0.0)),
            conviction=float(getattr(signal, "conviction", 0.0)),
            urgency=float(getattr(signal, "urgency", 0.0)),
            holding_period=str(getattr(signal, "holding_period", "days")),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_idx = i

    return best_idx

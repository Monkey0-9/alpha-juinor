import pytest
import pandas as pd
import shutil
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

collect_ignore = ["test_output.txt", "test_output_2.txt", "test_output_utf8.txt", "test_output_2_utf8.txt"]
if importlib.util.find_spec("torch") is None:
    collect_ignore.append("test_airllm.py")

@pytest.fixture
def mock_ohlcv_data():
    dates = pd.date_range(start="2023-01-01", periods=10, freq="B")
    data = {
        "Open": [100.0] * 10,
        "High": [105.0] * 10,
        "Low": [95.0] * 10,
        "Close": [102.0] * 10,
        "Volume": [1000] * 10
    }
    df = pd.DataFrame(data, index=dates)
    return df

@pytest.fixture
def temp_data_dir():
    path = Path("tests/temp_data")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    yield str(path)
    if path.exists():
        shutil.rmtree(path)

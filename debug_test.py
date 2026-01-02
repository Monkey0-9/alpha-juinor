
import sys
import os
import shutil
import pytest
from pathlib import Path

# Add root to path
sys.path.insert(0, os.getcwd())

from tests.test_backtester import test_backtester_execution_pipeline

class MockTmpPath:
    def __init__(self, path):
        self.path = Path(path)
    
    def __truediv__(self, other):
        return self.path / other

    def __getattr__(self, name):
        return getattr(self.path, name)

try:
    tmp = Path("debug_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    
    print("Running test_backtester_execution_pipeline...")
    test_backtester_execution_pipeline(tmp)
    print("Test passed!")
except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()

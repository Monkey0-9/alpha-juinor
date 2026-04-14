
#!/usr/bin/env python3
"""
Quant Fund OS - Main Entry Point
Institutional-grade quantitative trading platform.
"""

import argparse
import sys
import os
from mini_quant_fund.core.engine.orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description="Quant Fund OS")
    parser.add_argument("--mode", type=str, choices=["paper", "live"], default="paper",
                        help="Trading mode: paper (default) or live")
    parser.add_argument("--headless", action="store_true", 
                        help="Run without terminal dashboard")
    parser.add_argument("--config", type=str, default="configs/golden_config.yaml",
                        help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Initialize the high-level coordinator
    orchestrator = Orchestrator(mode=args.mode, headless=args.headless)
    
    try:
        orchestrator.start()
    except Exception as e:
        print(f"CRITICAL: System failed to launch: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

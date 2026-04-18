"""
INSTITUTIONAL UPGRADE: Enterprise Trading Platform
===================================================
Nexus v0.3.0 - Institutional Edition
Designed to match top-tier firms: Jane Street, Citadel, Jump Trading, Optiver

CAPABILITIES:
- Multi-asset execution (equities, fixed income, crypto, derivatives, FX)
- Market making across 235+ venues
- Ultra-low latency order routing
- Decentralized strategy architecture
- Institutional risk framework
- Cloud-native auto-scaling
- Real-time compliance & monitoring
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.nexus.core.context import engine_context
from src.nexus.institutional.orchestrator import InstitutionalOrchestrator
from src.nexus.institutional.deployment import CloudDeploymentManager

__version__ = "0.3.0-institutional"
__title__ = "Nexus Institutional Trading Platform"

def initialize_institutional_platform(config_path: str = None):
    """Initialize enterprise trading platform."""
    engine_context.initialize(config_path or "config/production.yaml")
    logger = engine_context.get_logger("nexus.institutional")
    logger.info(f"Initializing {__title__} v{__version__}")
    
    # Initialize institutional orchestrator
    orchestrator = InstitutionalOrchestrator()
    return orchestrator

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description=f"{__title__} - {__version__}")
    parser.add_argument("--mode", choices=["backtest", "paper", "live", "market-making"],
                       default="backtest", help="Execution mode")
    parser.add_argument("--asset-class", choices=["equities", "fixed-income", "crypto", "derivatives", "fx", "multi"],
                       default="equities", help="Asset class")
    parser.add_argument("--venues", type=int, default=235, help="Number of venues")
    parser.add_argument("--cloud-deploy", action="store_true", help="Deploy to cloud")
    parser.add_argument("--config", type=str, help="Config file path")
    
    args = parser.parse_args()
    
    orchestrator = initialize_institutional_platform(args.config)
    
    # Configure based on arguments
    orchestrator.set_execution_mode(args.mode)
    orchestrator.set_asset_classes([args.asset_class])
    orchestrator.set_venue_count(args.venues)
    
    if args.cloud_deploy:
        cloud_mgr = CloudDeploymentManager()
        cloud_mgr.deploy_to_azure()
    
    orchestrator.start()

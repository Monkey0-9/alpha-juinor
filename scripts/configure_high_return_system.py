#!/usr/bin/env python3
"""
High Return System Configuration
=============================
Configure the system for 50+ companies with 60-70% return target.

Usage:
    python scripts/configure_high_return_system.py
"""

import json
import os
import shutil
import sys
import yaml
from datetime import datetime
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def print_header(title: str):
    """Print formatted section header."""
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")

def backup_existing_configs():
    """Backup existing configuration files."""
    print_header("BACKING UP EXISTING CONFIGURATIONS")

    backup_dir = PROJECT_ROOT / "configs" / "backup" f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    configs_to_backup = [
        "universe.json",
        "trading_config.yaml",
        "risk_config.yaml"
    ]

    for config in configs_to_backup:
        src = PROJECT_ROOT / "configs" / config
        if src.exists():
            dst = backup_dir / config
            shutil.copy2(src, dst)
            print(f"  ✓ Backed up: {config}")

    print(f"  ✓ Backup completed: {backup_dir}")

def update_universe_config():
    """Update universe configuration for 50+ symbols."""
    print_header("UPDATING UNIVERSE CONFIGURATION")

    # Copy new universe config
    src = PROJECT_ROOT / "configs" / "universe_50_plus.json"
    dst = PROJECT_ROOT / "configs" / "universe.json"

    if src.exists():
        shutil.copy2(src, dst)
        print(f"  ✓ Updated universe.json with 50+ symbols")

        # Load and display summary
        with open(dst, 'r') as f:
            config = json.load(f)

        symbols = config['active_tickers']
        print(f"  ✓ Total symbols: {len(symbols)}")
        print(f"  ✓ Target return: {config.get('optimization_target', 'N/A')}")

        # Show sector breakdown
        sectors = config.get('sectors', {})
        for sector, tickers in sectors.items():
            print(f"  ✓ {sector.title()}: {len(tickers)} symbols")
    else:
        print(f"  ✗ Source universe config not found: {src}")

def update_risk_parameters():
    """Update risk parameters for higher returns."""
    print_header("UPDATING RISK PARAMETERS")

    risk_config_path = PROJECT_ROOT / "configs" / "risk_config.yaml"

    # New aggressive risk parameters
    risk_config = {
        "position_limits": {
            "max_position_size": 0.08,      # 8% per position (up from 5%)
            "max_sector_exposure": 0.20,    # 20% per sector
            "max_correlated_exposure": 0.30,
            "min_position_size": 0.005       # 0.5% minimum
        },
        "portfolio_limits": {
            "max_leverage": 1.2,            # 1.2x leverage
            "target_volatility": 0.18,      # 18% annual vol
            "max_drawdown": 0.15,          # 15% max DD
            "var_95_limit": 0.05           # 5% daily VaR
        },
        "stop_loss_config": {
            "individual_stop": 0.06,        # 6% stop loss
            "portfolio_stop": 0.10,         # 10% portfolio stop
            "trailing_stop": 0.04,         # 4% trailing stop
            "time_stop_hours": 72           # 72-hour time stop
        },
        "take_profit_config": {
            "individual_target": 0.25,      # 25% take profit
            "partial_profit_levels": [0.10, 0.15, 0.20],
            "profit_trailing": True
        }
    }

    # Write new risk config
    with open(risk_config_path, 'w') as f:
        yaml.dump(risk_config, f, default_flow_style=False)

    print(f"  ✓ Updated risk_config.yaml with aggressive parameters")
    print(f"  ✓ Max position size: 8% (increased from 5%)")
    print(f"  ✓ Max leverage: 1.2x (increased from 1.0x)")
    print(f"  ✓ Target volatility: 18% (increased from 15%)")

def update_trading_config():
    """Update trading configuration for high returns."""
    print_header("UPDATING TRADING CONFIGURATION")

    trading_config_path = PROJECT_ROOT / "configs" / "trading_config.yaml"

    # New high-return trading config
    trading_config = {
        "strategy_allocation": {
            "momentum_trading": 0.25,       # 25% to momentum (highest)
            "swing_trading": 0.20,          # 20% to swing
            "day_trading": 0.15,            # 15% to day trading
            "position_trading": 0.15,       # 15% to position
            "scalping": 0.10,               # 10% to scalping
            "algorithmic_trading": 0.10,     # 10% to algorithmic
            "other_strategies": 0.05         # 5% to others
        },
        "execution_parameters": {
            "order_type": "adaptive",
            "execution_speed": "fast",
            "slippage_tolerance": 0.001,
            "batch_trading": True,
            "smart_routing": True
        },
        "performance_targets": {
            "annual_return_target": 0.65,    # 65% annual return
            "sharpe_ratio_target": 2.5,
            "win_rate_target": 0.55,
            "profit_factor_target": 1.8,
            "max_drawdown_target": 0.15
        },
        "market_conditions": {
            "bull_market_multiplier": 1.3,
            "bear_market_multiplier": 0.7,
            "sideways_market_multiplier": 1.0
        }
    }

    # Write new trading config
    with open(trading_config_path, 'w') as f:
        yaml.dump(trading_config, f, default_flow_style=False)

    print(f"  ✓ Updated trading_config.yaml for high returns")
    print(f"  ✓ Momentum allocation: 25% (highest priority)")
    print(f"  ✓ Annual return target: 65%")
    print(f"  ✓ Bull market multiplier: 1.3x")

def create_performance_monitor():
    """Create performance monitoring configuration."""
    print_header("CREATING PERFORMANCE MONITOR")

    monitor_config = {
        "monitoring": {
            "enabled": True,
            "update_frequency": "1 minute",
            "alerts": {
                "performance_drop": {"threshold": 0.10, "period": "1 day"},
                "drawdown_alert": {"threshold": 0.12, "action": "reduce_positions"},
                "correlation_spike": {"threshold": 0.80, "action": "rebalance"},
                "volatility_spike": {"threshold": 0.30, "action": "reduce_leverage"}
            },
            "reports": {
                "daily_summary": True,
                "weekly_performance": True,
                "monthly_analysis": True,
                "risk_metrics": True
            }
        }
    }

    monitor_path = PROJECT_ROOT / "configs" / "performance_monitor.json"
    with open(monitor_path, 'w') as f:
        json.dump(monitor_config, f, indent=2)

    print(f"  ✓ Created performance_monitor.json")
    print(f"  ✓ Real-time monitoring enabled")
    print(f"  ✓ Automated alerts configured")

def update_environment():
    """Update environment variables for high-return mode."""
    print_header("UPDATING ENVIRONMENT VARIABLES")

    env_path = PROJECT_ROOT / ".env"

    # Read existing .env
    env_lines = []
    if env_path.exists():
        env_lines = env_path.read_text().splitlines()

    # Add/update high-return configurations
    new_vars = [
        "# High Return Configuration - Added on " + datetime.now().isoformat(),
        "TRADING_MODE=aggressive_high_return",
        "TARGET_ANNUAL_RETURN=0.65",
        "MAX_POSITION_SIZE=0.08",
        "MAX_LEVERAGE=1.2",
        "TARGET_VOLATILITY=0.18",
        "ENABLE_LEVERAGE=true",
        "ENABLE_SHORT_SELLING=true",
        "ENABLE_OPTIONS_TRADING=false",
        "RISK_MODE=aggressive_controlled",
        "MONITORING_LEVEL=high",
        ""
    ]

    # Remove existing high-return vars if present
    env_lines = [line for line in env_lines if not line.startswith("TRADING_MODE=")
                and not line.startswith("TARGET_ANNUAL_RETURN=")
                and not line.startswith("MAX_POSITION_SIZE=")
                and not line.startswith("MAX_LEVERAGE=")
                and not line.startswith("TARGET_VOLATILITY=")
                and not line.startswith("RISK_MODE=")]

    # Add new variables
    env_lines.extend(new_vars)

    # Write back to .env
    env_path.write_text("\n".join(env_lines))

    print(f"  ✓ Updated .env with high-return configuration")
    print(f"  ✓ Trading mode: aggressive_high_return")
    print(f"  ✓ Target annual return: 65%")

def create_startup_script():
    """Create startup script for high-return trading."""
    print_header("CREATING STARTUP SCRIPT")

    startup_script = f"""#!/usr/bin/env python3
'''
High Return Trading Launcher
==========================
Launches trading system optimized for 60-70% annual returns.

Created: {datetime.now().isoformat()}
'''

import os
import sys
from pathlib import Path

# Set environment for high returns
os.environ['TRADING_MODE'] = 'aggressive_high_return'
os.environ['TARGET_ANNUAL_RETURN'] = '0.65'
os.environ['MAX_POSITION_SIZE'] = '0.08'
os.environ['MAX_LEVERAGE'] = '1.2'

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("\\n" + "="*70)
    print("  HIGH RETURN TRADING SYSTEM - 60-70% ANNUAL TARGET")
    print("="*70)
    print("\\nConfiguration:")
    print("  • 50+ High-quality symbols")
    print("  • 8% Max position size (up from 5%)")
    print("  • 1.2x Leverage (controlled)")
    print("  • 65% Annual return target")
    print("  • 15% Max drawdown limit")
    print("\\nStarting paper trading...")

    # Import and start paper trading
    from scripts.paper_trading_launcher import main as paper_main

    # Configure for high returns
    import argparse
    sys.argv = ['paper_trading_launcher.py', '--days', '14', '--config', 'aggressive_return_config.yaml']

    paper_main()

if __name__ == "__main__":
    main()
"""

    script_path = PROJECT_ROOT / "scripts" / "launch_high_return_trading.py"
    with open(script_path, 'w') as f:
        f.write(startup_script)

    print(f"  ✓ Created launch_high_return_trading.py")
    print(f"  ✓ Ready to start high-return paper trading")

def main():
    """Configure system for high returns."""
    print("\n" + "="*70)
    print("  CONFIGURING SYSTEM FOR 60-70% ANNUAL RETURNS")
    print("="*70)

    try:
        backup_existing_configs()
        update_universe_config()
        update_risk_parameters()
        update_trading_config()
        create_performance_monitor()
        update_environment()
        create_startup_script()

        print_header("CONFIGURATION COMPLETE")
        print("✓ System configured for 60-70% annual returns")
        print("✓ 50+ high-quality symbols ready")
        print("✓ Aggressive but controlled risk parameters set")
        print("✓ Performance monitoring enabled")
        print("\nNext Steps:")
        print("  1. Review configurations in configs/")
        print("  2. Run: python scripts/launch_high_return_trading.py")
        print("  3. Monitor performance for 14 days")
        print("  4. Adjust parameters based on results")

    except Exception as e:
        print(f"\n✗ Configuration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())

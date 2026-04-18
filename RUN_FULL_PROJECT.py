#!/usr/bin/env python3
"""
NEXUS INSTITUTIONAL v0.3.0 - COMPLETE EXECUTION GUIDE
=====================================================
This script demonstrates how to run the project fully in all modes.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def show_quick_start():
    """Show quick start commands."""
    print_section("QUICK START: MOST COMMON COMMANDS")
    
    commands = [
        ("1. Run Full Backtest", "python run_institutional_backtest.py"),
        ("2. Start Development Engine", "python main.py --mode sim"),
        ("3. Run Verification Tests", "python verify_institutional_system.py"),
        ("4. Launch Institutional Platform", 
         "python nexus_institutional.py --mode backtest --asset-class multi --venues 235"),
        ("5. Test Market Making", 
         "python nexus_institutional.py --mode market-making --asset-class equities --venues 50"),
    ]
    
    for title, cmd in commands:
        print(f"{title}")
        print(f"  $ {cmd}\n")

def show_modes():
    """Show all execution modes."""
    print_section("EXECUTION MODES")
    
    modes = {
        "BACKTEST": {
            "description": "Historical data simulation",
            "use_case": "Strategy research, performance analysis",
            "command": "python run_institutional_backtest.py"
        },
        "PAPER": {
            "description": "Paper trading (no real orders)",
            "use_case": "System testing, live market data without risk",
            "command": "python nexus_institutional.py --mode paper --asset-class equities"
        },
        "LIVE": {
            "description": "Live trading (real orders)",
            "use_case": "Production trading (requires real credentials)",
            "command": "python nexus_institutional.py --mode live --asset-class multi"
        },
        "MARKET-MAKING": {
            "description": "Market making strategies",
            "use_case": "Liquidity provision across venues",
            "command": "python nexus_institutional.py --mode market-making --asset-class equities"
        }
    }
    
    for mode, details in modes.items():
        print(f"{mode}")
        print(f"  Description:  {details['description']}")
        print(f"  Use Case:     {details['use_case']}")
        print(f"  Command:      {details['command']}\n")

def show_asset_classes():
    """Show available asset classes."""
    print_section("ASSET CLASSES")
    
    classes = {
        "equities": {
            "venues": "NYSE, NASDAQ, LSE, Euronext, etc.",
            "example": "python nexus_institutional.py --mode backtest --asset-class equities"
        },
        "fixed-income": {
            "venues": "Rates, bonds, swaps markets",
            "example": "python nexus_institutional.py --mode backtest --asset-class fixed-income"
        },
        "crypto": {
            "venues": "Kraken, Coinbase, Binance, OKX, FTX",
            "example": "python nexus_institutional.py --mode backtest --asset-class crypto"
        },
        "derivatives": {
            "venues": "CME, CBOT, CBOE, etc.",
            "example": "python nexus_institutional.py --mode backtest --asset-class derivatives"
        },
        "fx": {
            "venues": "EBS, Reuters, Bloomberg, FXall",
            "example": "python nexus_institutional.py --mode backtest --asset-class fx"
        },
        "multi": {
            "venues": "ALL asset classes combined (6 total)",
            "example": "python nexus_institutional.py --mode backtest --asset-class multi"
        }
    }
    
    for ac, details in classes.items():
        print(f"{ac.upper()}")
        print(f"  Venues:  {details['venues']}")
        print(f"  Example: {details['example']}\n")

def show_venue_options():
    """Show venue configuration."""
    print_section("VENUE CONFIGURATION")
    
    print("Number of Venues to Route Through:\n")
    
    options = [
        ("10", "Major venues (NYSE, NASDAQ, CBOE, LSE, etc.)"),
        ("50", "Regional + international venues"),
        ("100", "Extended global coverage"),
        ("235", "MAXIMUM - All supported venues (Jane Street, Citadel, HRT level)")
    ]
    
    for count, description in options:
        print(f"--venues {count:3}  →  {description}\n")
    
    print("Example: python nexus_institutional.py --mode backtest --venues 235\n")

def show_development_workflow():
    """Show typical development workflow."""
    print_section("DEVELOPMENT WORKFLOW")
    
    print("""
Step 1: VERIFY SYSTEM IS OPERATIONAL
  $ python verify_institutional_system.py
  Expected: 7/7 tests PASSED ✅
  
Step 2: RUN BACKTEST TO VALIDATE BASELINE
  $ python run_institutional_backtest.py
  Result: Performance metrics (Sharpe, drawdown, etc.)
  
Step 3: START DEVELOPMENT ENGINE
  $ python main.py --mode sim
  This runs market simulation mode (no real trades)
  
Step 4: TEST SPECIFIC CONFIGURATION
  $ python nexus_institutional.py --mode backtest --asset-class equities --venues 50
  
Step 5: TEST MARKET MAKING
  $ python nexus_institutional.py --mode market-making --asset-class equities
  
Step 6: FINAL VERIFICATION BEFORE LIVE
  $ python verify_institutional_system.py
    """)

def show_advanced_commands():
    """Show advanced command combinations."""
    print_section("ADVANCED COMMAND COMBINATIONS")
    
    commands = [
        {
            "name": "Institutional Research Setup",
            "cmd": "python nexus_institutional.py --mode backtest --asset-class multi --venues 235",
            "notes": "Full institutional backtest with all venues and asset classes"
        },
        {
            "name": "Equities Only (Fast)",
            "cmd": "python nexus_institutional.py --mode backtest --asset-class equities --venues 50",
            "notes": "Faster execution focusing on equity markets"
        },
        {
            "name": "Market Making Testing",
            "cmd": "python nexus_institutional.py --mode market-making --asset-class equities",
            "notes": "Test market making strategies on equity venues"
        },
        {
            "name": "Crypto Research",
            "cmd": "python nexus_institutional.py --mode backtest --asset-class crypto --venues 5",
            "notes": "Focus on cryptocurrency markets"
        },
        {
            "name": "Multi-Asset Fund Simulation",
            "cmd": "python nexus_institutional.py --mode backtest --asset-class multi --venues 100",
            "notes": "Simulate institutional fund across 100 venues"
        },
        {
            "name": "Paper Trading (Live Market Data, No Risk)",
            "cmd": "python nexus_institutional.py --mode paper --asset-class multi",
            "notes": "Safe mode with live data but no actual orders"
        }
    ]
    
    for cmd_info in commands:
        print(f"{cmd_info['name'].upper()}")
        print(f"  Command: {cmd_info['cmd']}")
        print(f"  Notes:   {cmd_info['notes']}\n")

def show_production_deployment():
    """Show production deployment steps."""
    print_section("PRODUCTION DEPLOYMENT ON AZURE")
    
    print("""
STEP 1: PREREQUISITES
  ✓ Azure CLI installed
  ✓ Terraform >= 1.0 installed
  ✓ kubectl configured
  ✓ Docker installed
  
STEP 2: SETUP TERRAFORM STATE
  $ az storage account create \\
      --name nexusterraformstate \\
      --resource-group nexus-terraform-state \\
      --location westus2 \\
      --sku Standard_LRS
  
  $ az storage container create \\
      --name tfstate \\
      --account-name nexusterraformstate

STEP 3: DEPLOY INFRASTRUCTURE
  $ cd infrastructure/terraform
  $ terraform init
  $ terraform plan -out=tfplan
  $ terraform apply tfplan

STEP 4: BUILD & PUSH DOCKER IMAGE
  $ docker build -t nexus:0.3.0-institutional .
  $ docker tag nexus:0.3.0-institutional nexustrading.azurecr.io/nexus:latest
  $ az acr login --name nexustrading
  $ docker push nexustrading.azurecr.io/nexus:latest

STEP 5: DEPLOY TO AKS
  $ az aks get-credentials \\
      --resource-group nexus-trading-rg \\
      --name nexus-aks-production
  
  $ kubectl apply -f infrastructure/kubernetes/
  $ kubectl rollout status deployment/nexus

STEP 6: ACCESS PLATFORM
  $ kubectl port-forward svc/nexus 8080:8080
  $ open http://localhost:8080
    """)

def show_configuration_guide():
    """Show configuration options."""
    print_section("CONFIGURATION GUIDE")
    
    print("""
DEVELOPMENT CONFIG (config/development.yaml)
  - Used for: Local testing, backtesting
  - Initial cash: $100,000
  - Commission: 1.0 bps
  - Slippage: 0.1
  
PRODUCTION CONFIG (config/production.yaml)
  - Used for: Live trading, institutional operations
  - Initial cash: $1,000,000,000 ($1B)
  - Commission: 0.5 bps (negotiated rates)
  - Multi-asset support with all venues

OVERRIDE ENVIRONMENT VARIABLES
  export DATABASE_HOST=...
  export REDIS_HOST=...
  export KAFKA_BROKER_1=...
  export TRADING_ENABLED=true  (for live mode)
    """)

def show_monitoring():
    """Show monitoring and observability."""
    print_section("MONITORING & OBSERVABILITY")
    
    print("""
REAL-TIME MONITORING
  Dashboard:  http://localhost:8080/dashboard
  Metrics:    http://localhost:8080/metrics
  Logs:       kubernetes logs
  
AZURE MONITORING
  Application Insights:  Azure Portal → Insights
  Monitor:               Azure Monitor → Metrics
  Alerts:                Set up alerts for:
                         - Daily loss > $500k
                         - Order latency p99 > 10ms
                         - Fill rate < 85%
  
CHECK LOGS
  $ kubectl logs deployment/nexus -f
  $ kubectl logs -l app=nexus --tail=100
  
PERFORMANCE METRICS
  Order Latency (p99):      < 100 microseconds
  Throughput:               > 100k orders/day
  Memory Usage:             < 256 MB baseline
  CPU Utilization:          < 70% under normal load
    """)

def show_troubleshooting():
    """Show troubleshooting common issues."""
    print_section("TROUBLESHOOTING")
    
    print("""
ISSUE: "ModuleNotFoundError: No module named 'nexus'"
  FIX: Make sure src is in PYTHONPATH
  $ export PYTHONPATH="${PYTHONPATH}:/path/to/mini-quant-fund/src"

ISSUE: "Database connection failed"
  FIX: Check DATABASE_HOST environment variable
  $ echo $DATABASE_HOST
  Or use development.yaml which doesn't require database

ISSUE: "Venue configuration parsing error"
  FIX: Validate production.yaml syntax
  $ python -m yaml config/production.yaml

ISSUE: "Backtest runs slowly"
  SOLUTION: Reduce number of venues or use fewer asset classes
  $ python nexus_institutional.py --mode backtest --asset-class equities --venues 20

ISSUE: "Out of memory error"
  SOLUTION: Enable memory-mapped files in config
  Edit config/production.yaml:
    low_latency:
      use_memory_mapping: true

ISSUE: Tests failing after code changes
  FIX: Run verification suite
  $ python verify_institutional_system.py
    """)

def show_next_steps():
    """Show recommended next steps."""
    print_section("RECOMMENDED NEXT STEPS")
    
    print("""
1. RUN VERIFICATION TESTS
   $ python verify_institutional_system.py
   Expected Result: 7/7 tests PASSED ✅

2. RUN BASELINE BACKTEST
   $ python run_institutional_backtest.py
   Expected Result: Sharpe 0.05-0.56, Results JSON file

3. EXPLORE CONFIGURATIONS
   $ python nexus_institutional.py --mode backtest --asset-class multi --venues 235
   Expected Result: Platform initializes with all components

4. TEST MARKET MAKING
   $ python nexus_institutional.py --mode market-making --asset-class equities
   Expected Result: Market making engine ready

5. READ DOCUMENTATION
   • INSTITUTIONAL_COMPLETION_REPORT.md - Full technical overview
   • INSTITUTIONAL_DEPLOYMENT_GUIDE.md - Deployment runbook
   • config/production.yaml - Configuration reference

6. PREPARE FOR PRODUCTION
   • Set up Azure account
   • Configure API keys in Key Vault
   • Review risk parameters
   • Set up monitoring alerts

7. DEPLOY TO CLOUD
   • Run Terraform scripts
   • Push Docker image to ACR
   • Deploy to AKS
   • Verify in staging environment

8. GO LIVE
   • Enable trading_enabled: true
   • Monitor for 24 hours
   • Scale as needed
    """)

def main():
    """Main execution guide."""
    print("\n" + "="*80)
    print("  NEXUS INSTITUTIONAL v0.3.0 - COMPLETE PROJECT EXECUTION GUIDE")
    print("  How to Run the Full Advanced Trading Platform")
    print("="*80)
    
    # Show all guides
    show_quick_start()
    show_modes()
    show_asset_classes()
    show_venue_options()
    show_development_workflow()
    show_advanced_commands()
    show_configuration_guide()
    show_monitoring()
    show_production_deployment()
    show_troubleshooting()
    show_next_steps()
    
    print("\n" + "="*80)
    print("  START HERE: python verify_institutional_system.py")
    print("  Then: python run_institutional_backtest.py")
    print("  Then: python main.py --mode sim")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

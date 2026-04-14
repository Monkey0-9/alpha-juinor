#!/usr/bin/env python3
"""
DIAGNOSE TRADING EXECUTION - CRITICAL SETUP VERIFICATION
=========================================================

Enterprise-grade diagnostic script for MiniQuantFund trading system.
Validates all critical components before trading begins.

Usage:
    python diagnose_trading_execution.py
    python diagnose_trading_execution.py --fix  # Auto-fix issues where possible

Exit Codes:
    0 - All systems ready for trading
    1 - Critical issues found (do not trade)
    2 - Warnings only (trading possible but suboptimal)
"""

import sys
import os
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/diagnose_trading.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


class CheckStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class DiagnosticCheck:
    name: str
    status: CheckStatus
    message: str
    details: Dict = None
    fixable: bool = False

    def to_dict(self):
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details or {},
            "fixable": self.fixable
        }


class TradingSystemDiagnostics:
    """Comprehensive trading system diagnostic engine."""
    
    def __init__(self, auto_fix: bool = False):
        self.auto_fix = auto_fix
        self.checks: List[DiagnosticCheck] = []
        self.project_root = Path(__file__).parent
        self.src_path = self.project_root / "src" / "mini_quant_fund"
        
    def run_all_checks(self) -> Tuple[int, int, int]:
        """Run all diagnostic checks and return (passed, failed, warnings)."""
        logger.info("=" * 80)
        logger.info("MINIQUANTFUND TRADING SYSTEM DIAGNOSTICS")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {datetime.utcnow().isoformat()}Z")
        logger.info(f"Project Root: {self.project_root}")
        logger.info("")
        
        # Phase 1: Environment & Configuration
        self._check_environment_variables()
        self._check_virtual_environment()
        self._check_python_version()
        self._check_dependencies()
        
        # Phase 2: Critical Files & Structure
        self._check_project_structure()
        self._check_critical_files()
        self._check_runtime_directories()
        
        # Phase 3: Trading Components
        self._check_execute_trades_flag()
        self._check_kill_switch_status()
        self._check_trading_mode()
        self._check_broker_configuration()
        
        # Phase 4: Data & Execution Pipeline
        self._check_data_router()
        self._check_execution_handler()
        self._check_strategy_components()
        
        # Phase 5: Safety & Risk Management
        self._check_circuit_breaker()
        self._check_risk_gates()
        self._check_governance_system()
        
        # Phase 6: Infrastructure
        self._check_database_connection()
        self._check_monitoring_stack()
        self._check_ml_pipeline()
        
        # Phase 7: Performance
        self._check_latency_benchmarks()
        self._check_throughput_capabilities()
        
        return self._summarize_results()
    
    def _check_environment_variables(self):
        """Check critical environment variables."""
        critical_vars = [
            "ALPACA_API_KEY",
            "ALPACA_SECRET_KEY"
        ]
        
        optional_vars = [
            "ALPACA_BASE_URL",
            "CIRCUIT_STATE_PATH",
            "DATABASE_URL"
        ]
        
        missing_critical = []
        for var in critical_vars:
            if not os.getenv(var):
                missing_critical.append(var)
        
        if missing_critical:
            self.checks.append(DiagnosticCheck(
                name="Environment Variables (Critical)",
                status=CheckStatus.WARN if os.getenv("TRADING_MODE", "paper") == "paper" else CheckStatus.FAIL,
                message=f"Missing critical env vars: {', '.join(missing_critical)}",
                details={"missing": missing_critical},
                fixable=False
            ))
        else:
            self.checks.append(DiagnosticCheck(
                name="Environment Variables (Critical)",
                status=CheckStatus.PASS,
                message="All critical environment variables configured",
                details={"configured": critical_vars}
            ))
        
        # Check optional vars
        missing_optional = [var for var in optional_vars if not os.getenv(var)]
        if missing_optional:
            self.checks.append(DiagnosticCheck(
                name="Environment Variables (Optional)",
                status=CheckStatus.WARN,
                message=f"Missing optional vars (will use defaults): {', '.join(missing_optional)}",
                details={"missing": missing_optional}
            ))
        else:
            self.checks.append(DiagnosticCheck(
                name="Environment Variables (Optional)",
                status=CheckStatus.PASS,
                message="All optional environment variables configured"
            ))
    
    def _check_virtual_environment(self):
        """Check if running in virtual environment."""
        in_venv = (
            hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        )
        
        self.checks.append(DiagnosticCheck(
            name="Virtual Environment",
            status=CheckStatus.PASS if in_venv else CheckStatus.WARN,
            message="Running in virtual environment" if in_venv else "Not in virtual environment (recommended)",
            fixable=False
        ))
    
    def _check_python_version(self):
        """Check Python version compatibility."""
        version = sys.version_info
        min_version = (3, 9)
        
        if version >= min_version:
            self.checks.append(DiagnosticCheck(
                name="Python Version",
                status=CheckStatus.PASS,
                message=f"Python {version.major}.{version.minor}.{version.micro} (>= 3.9 required)",
                details={"version": f"{version.major}.{version.minor}.{version.micro}"}
            ))
        else:
            self.checks.append(DiagnosticCheck(
                name="Python Version",
                status=CheckStatus.FAIL,
                message=f"Python {version.major}.{version.minor} < 3.9 required",
                fixable=False
            ))
    
    def _check_dependencies(self):
        """Check critical Python dependencies."""
        required_packages = [
            "pandas", "numpy", "scipy", "scikit-learn",
            "yfinance", "requests", "sqlalchemy"
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            self.checks.append(DiagnosticCheck(
                name="Python Dependencies",
                status=CheckStatus.FAIL,
                message=f"Missing packages: {', '.join(missing)}",
                details={"missing": missing, "install_cmd": f"pip install {' '.join(missing)}"},
                fixable=True
            ))
        else:
            self.checks.append(DiagnosticCheck(
                name="Python Dependencies",
                status=CheckStatus.PASS,
                message=f"All {len(required_packages)} critical packages installed"
            ))
    
    def _check_project_structure(self):
        """Validate src/mini_quant_fund modular structure."""
        required_dirs = [
            "core", "data", "execution", "strategies",
            "risk", "brokers", "governance", "safety"
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = self.src_path / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            self.checks.append(DiagnosticCheck(
                name="Project Structure",
                status=CheckStatus.FAIL,
                message=f"Missing directories: {', '.join(missing_dirs)}",
                fixable=False
            ))
        else:
            self.checks.append(DiagnosticCheck(
                name="Project Structure",
                status=CheckStatus.PASS,
                message="All required modules present in src/mini_quant_fund/",
                details={"modules": required_dirs}
            ))
    
    def _check_critical_files(self):
        """Check critical configuration and executable files."""
        critical_files = [
            ("main.py", "Main entry point"),
            ("pyproject.toml", "Package configuration"),
            ("src/mini_quant_fund/configs/golden_config.yaml", "Golden config"),
            (".env.example", "Environment template"),
        ]
        
        missing_files = []
        for file_path, description in critical_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(f"{file_path} ({description})")
        
        if missing_files:
            self.checks.append(DiagnosticCheck(
                name="Critical Files",
                status=CheckStatus.FAIL,
                message=f"Missing: {', '.join(missing_files)}",
                fixable=False
            ))
        else:
            self.checks.append(DiagnosticCheck(
                name="Critical Files",
                status=CheckStatus.PASS,
                message="All critical files present"
            ))
    
    def _check_runtime_directories(self):
        """Ensure runtime directories exist."""
        required_dirs = ["logs", "runtime", "data", "output"]
        created_dirs = []
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                if self.auto_fix:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(dir_name)
                else:
                    self.checks.append(DiagnosticCheck(
                        name=f"Runtime Directory: {dir_name}",
                        status=CheckStatus.WARN,
                        message=f"Directory missing: {dir_name}/",
                        fixable=True
                    ))
        
        if created_dirs:
            self.checks.append(DiagnosticCheck(
                name="Runtime Directories",
                status=CheckStatus.PASS,
                message=f"Created directories: {', '.join(created_dirs)}",
                details={"created": created_dirs}
            ))
        elif not any(c.name.startswith("Runtime Directory:") for c in self.checks):
            self.checks.append(DiagnosticCheck(
                name="Runtime Directories",
                status=CheckStatus.PASS,
                message="All runtime directories present"
            ))
    
    def _check_execute_trades_flag(self):
        """Check EXECUTE_TRADES environment flag."""
        execute_trades = os.getenv("EXECUTE_TRADES", "false").lower()
        
        if execute_trades in ("true", "1", "yes"):
            self.checks.append(DiagnosticCheck(
                name="EXECUTE_TRADES Flag",
                status=CheckStatus.PASS,
                message="EXECUTE_TRADES enabled: True",
                details={"value": execute_trades}
            ))
        else:
            self.checks.append(DiagnosticCheck(
                name="EXECUTE_TRADES Flag",
                status=CheckStatus.WARN,
                message="EXECUTE_TRADES disabled (set to 'true' to enable trading)",
                details={"current_value": execute_trades, "set_command": "set EXECUTE_TRADES=true"},
                fixable=True
            ))
    
    def _check_kill_switch_status(self):
        """Check kill switch file status."""
        kill_switch_path = self.project_root / "runtime" / "KILL_SWITCH"
        
        if kill_switch_path.exists():
            self.checks.append(DiagnosticCheck(
                name="Kill Switch Status",
                status=CheckStatus.FAIL,
                message="KILL_SWITCH file detected - trading is PAUSED",
                details={"path": str(kill_switch_path), "remove_command": "del runtime/KILL_SWITCH"},
                fixable=True
            ))
        else:
            self.checks.append(DiagnosticCheck(
                name="Kill Switch Status",
                status=CheckStatus.PASS,
                message="Kill switch not engaged - trading allowed"
            ))
    
    def _check_trading_mode(self):
        """Check trading mode configuration."""
        mode = os.getenv("TRADING_MODE", "paper").lower()
        
        if mode in ("paper", "live"):
            self.checks.append(DiagnosticCheck(
                name="Trading Mode",
                status=CheckStatus.PASS,
                message=f"TRADING_MODE={mode}",
                details={"mode": mode, "paper_warning": "Paper mode uses virtual money" if mode == "paper" else "LIVE MODE - REAL MONEY"}
            ))
        else:
            self.checks.append(DiagnosticCheck(
                name="Trading Mode",
                status=CheckStatus.FAIL,
                message=f"Invalid TRADING_MODE: {mode} (must be 'paper' or 'live')",
                fixable=True
            ))
    
    def _check_broker_configuration(self):
        """Verify broker configurations are accessible."""
        try:
            sys.path.insert(0, str(self.project_root / "src"))
            from mini_quant_fund.brokers.alpaca_broker import AlpacaBroker
            from mini_quant_fund.brokers.mock_broker import MockBroker
            
            brokers_available = {
                "AlpacaBroker": True,
                "MockBroker": True
            }
            
            # Test mock broker instantiation
            try:
                mock = MockBroker()
                brokers_available["MockBroker"] = True
            except Exception as e:
                brokers_available["MockBroker"] = str(e)
            
            self.checks.append(DiagnosticCheck(
                name="Broker Configuration",
                status=CheckStatus.PASS,
                message="Brokers module available",
                details=brokers_available
            ))
        except Exception as e:
            self.checks.append(DiagnosticCheck(
                name="Broker Configuration",
                status=CheckStatus.FAIL,
                message=f"Failed to load broker modules: {e}",
                fixable=False
            ))
    
    def _check_data_router(self):
        """Verify DataRouter is functional."""
        try:
            sys.path.insert(0, str(self.project_root / "src"))
            from mini_quant_fund.data.collectors.data_router import DataRouter
            
            router = DataRouter()
            
            self.checks.append(DiagnosticCheck(
                name="Data Router",
                status=CheckStatus.PASS,
                message="DataRouter initialized successfully",
                details={
                    "providers": list(router.providers.keys()),
                    "provider_count": len(router.providers)
                }
            ))
        except Exception as e:
            self.checks.append(DiagnosticCheck(
                name="Data Router",
                status=CheckStatus.FAIL,
                message=f"DataRouter failed: {e}",
                fixable=False
            ))
    
    def _check_execution_handler(self):
        """Verify execution handler components."""
        try:
            sys.path.insert(0, str(self.project_root / "src"))
            from mini_quant_fund.execution.alpaca_handler import AlpacaExecutionHandler
            
            # Check if handler can be instantiated (may fail without API keys)
            handler = AlpacaExecutionHandler(paper=True)
            
            self.checks.append(DiagnosticCheck(
                name="Execution Handler",
                status=CheckStatus.PASS,
                message="AlpacaExecutionHandler available",
                details={"paper_mode": handler.base_url}
            ))
        except Exception as e:
            self.checks.append(DiagnosticCheck(
                name="Execution Handler",
                status=CheckStatus.WARN,
                message=f"ExecutionHandler loaded but may need configuration: {e}",
                fixable=False
            ))
    
    def _check_strategy_components(self):
        """Verify strategy components are loadable."""
        try:
            sys.path.insert(0, str(self.project_root / "src"))
            from mini_quant_fund.strategies.institutional_strategy import InstitutionalStrategy
            
            strategy = InstitutionalStrategy()
            
            self.checks.append(DiagnosticCheck(
                name="Strategy Components",
                status=CheckStatus.PASS,
                message="InstitutionalStrategy loaded",
                details={"strategy_type": "institutional"}
            ))
        except Exception as e:
            self.checks.append(DiagnosticCheck(
                name="Strategy Components",
                status=CheckStatus.FAIL,
                message=f"Strategy load failed: {e}",
                fixable=False
            ))
    
    def _check_circuit_breaker(self):
        """Verify circuit breaker is functional."""
        try:
            sys.path.insert(0, str(self.project_root / "src"))
            from mini_quant_fund.safety.circuit_breaker import CircuitBreaker, CircuitConfig
            
            cb = CircuitBreaker()
            state = cb.get_state()
            
            self.checks.append(DiagnosticCheck(
                name="Circuit Breaker",
                status=CheckStatus.PASS if not cb.is_halted() else CheckStatus.FAIL,
                message=f"Circuit breaker {'HALTED' if cb.is_halted() else 'Active'}",
                details={
                    "halted": cb.is_halted(),
                    "daily_pnl": state.get("daily_pnl_usd", 0),
                    "weekly_pnl": state.get("weekly_pnl_usd", 0)
                },
                fixable=cb.is_halted()
            ))
        except Exception as e:
            self.checks.append(DiagnosticCheck(
                name="Circuit Breaker",
                status=CheckStatus.WARN,
                message=f"Circuit breaker check failed: {e}",
                fixable=False
            ))
    
    def _check_risk_gates(self):
        """Verify risk management gates."""
        try:
            sys.path.insert(0, str(self.project_root / "src"))
            from mini_quant_fund.governance.lifecycle_manager import LifecycleManager
            
            lm = LifecycleManager()
            
            self.checks.append(DiagnosticCheck(
                name="Risk Gates",
                status=CheckStatus.PASS,
                message="LifecycleManager (risk gates) available",
                details={"gates": "initialized"}
            ))
        except Exception as e:
            self.checks.append(DiagnosticCheck(
                name="Risk Gates",
                status=CheckStatus.WARN,
                message=f"Risk gates check: {e}",
                fixable=False
            ))
    
    def _check_governance_system(self):
        """Verify governance components."""
        try:
            sys.path.insert(0, str(self.project_root / "src"))
            from mini_quant_fund.governance.lifecycle_manager import LifecycleManager
            
            self.checks.append(DiagnosticCheck(
                name="Governance System",
                status=CheckStatus.PASS,
                message="Governance modules available"
            ))
        except Exception as e:
            self.checks.append(DiagnosticCheck(
                name="Governance System",
                status=CheckStatus.WARN,
                message=f"Governance check: {e}"
            ))
    
    def _check_database_connection(self):
        """Check database connectivity."""
        try:
            db_path = self.project_root / "mini_quant.db"
            if db_path.exists():
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 5")
                tables = cursor.fetchall()
                conn.close()
                
                self.checks.append(DiagnosticCheck(
                    name="Database Connection",
                    status=CheckStatus.PASS,
                    message=f"SQLite database connected ({len(tables)} tables)",
                    details={"tables": [t[0] for t in tables]}
                ))
            else:
                self.checks.append(DiagnosticCheck(
                    name="Database Connection",
                    status=CheckStatus.WARN,
                    message="Database file not found (will be created on first run)",
                    fixable=False
                ))
        except Exception as e:
            self.checks.append(DiagnosticCheck(
                name="Database Connection",
                status=CheckStatus.WARN,
                message=f"Database check: {e}"
            ))
    
    def _check_monitoring_stack(self):
        """Verify monitoring infrastructure."""
        try:
            sys.path.insert(0, str(self.project_root / "src"))
            from mini_quant_fund.monitoring.profiler import LatencyProfiler
            
            self.checks.append(DiagnosticCheck(
                name="Monitoring Stack",
                status=CheckStatus.PASS,
                message="Monitoring components available"
            ))
        except Exception as e:
            self.checks.append(DiagnosticCheck(
                name="Monitoring Stack",
                status=CheckStatus.WARN,
                message=f"Monitoring check: {e}"
            ))
    
    def _check_ml_pipeline(self):
        """Check ML pipeline status."""
        try:
            mlruns_path = self.project_root / "mlruns"
            
            if mlruns_path.exists():
                runs = list(mlruns_path.iterdir())
                self.checks.append(DiagnosticCheck(
                    name="ML Pipeline",
                    status=CheckStatus.PASS,
                    message=f"MLflow tracking available ({len(runs)} runs)",
                    details={"mlruns_path": str(mlruns_path)}
                ))
            else:
                self.checks.append(DiagnosticCheck(
                    name="ML Pipeline",
                    status=CheckStatus.WARN,
                    message="MLflow runs directory not found"
                ))
        except Exception as e:
            self.checks.append(DiagnosticCheck(
                name="ML Pipeline",
                status=CheckStatus.WARN,
                message=f"ML pipeline check: {e}"
            ))
    
    def _check_latency_benchmarks(self):
        """Check if latency benchmarks exist."""
        benchmark_files = [
            "src/mini_quant_fund/benchmarks/latency_async.py",
            "src/mini_quant_fund/benchmarks/latency_baseline.py"
        ]
        
        available = []
        for bf in benchmark_files:
            if (self.project_root / bf).exists():
                available.append(bf)
        
        if available:
            self.checks.append(DiagnosticCheck(
                name="Latency Benchmarks",
                status=CheckStatus.PASS,
                message=f"{len(available)} benchmark scripts available",
                details={"benchmarks": available}
            ))
        else:
            self.checks.append(DiagnosticCheck(
                name="Latency Benchmarks",
                status=CheckStatus.WARN,
                message="No latency benchmark scripts found"
            ))
    
    def _check_throughput_capabilities(self):
        """Check throughput test capabilities."""
        throughput_path = self.project_root / "benchmarks" / "throughput_test.py"
        
        if throughput_path.exists():
            self.checks.append(DiagnosticCheck(
                name="Throughput Test",
                status=CheckStatus.PASS,
                message="Throughput benchmark available",
                details={"path": str(throughput_path)}
            ))
        else:
            self.checks.append(DiagnosticCheck(
                name="Throughput Test",
                status=CheckStatus.WARN,
                message="throughput_test.py not found (create to prove 1000+ RPS claims)"
            ))
    
    def _summarize_results(self) -> Tuple[int, int, int]:
        """Summarize diagnostic results."""
        passed = sum(1 for c in self.checks if c.status == CheckStatus.PASS)
        failed = sum(1 for c in self.checks if c.status == CheckStatus.FAIL)
        warnings = sum(1 for c in self.checks if c.status == CheckStatus.WARN)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("DIAGNOSTIC RESULTS SUMMARY")
        logger.info("=" * 80)
        
        # Print detailed results
        for check in self.checks:
            icon = {
                CheckStatus.PASS: "✅",
                CheckStatus.FAIL: "❌",
                CheckStatus.WARN: "⚠️",
                CheckStatus.SKIP: "⏭️"
            }.get(check.status, "❓")
            
            logger.info(f"{icon} {check.name}: {check.message}")
        
        logger.info("")
        logger.info("-" * 80)
        logger.info(f"TOTAL: {passed} PASSED | {failed} FAILED | {warnings} WARNINGS")
        logger.info("-" * 80)
        
        if failed > 0:
            logger.info("STATUS: ❌ CRITICAL ISSUES FOUND - DO NOT TRADE")
            logger.info("Run with --fix flag to auto-fix issues where possible")
        elif warnings > 0:
            logger.info("STATUS: ⚠️ WARNINGS ONLY - TRADING POSSIBLE BUT SUBOPTIMAL")
        else:
            logger.info("STATUS: ✅ ALL SYSTEMS READY FOR TRADING")
        
        logger.info("=" * 80)
        
        # Save detailed report
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "total": len(self.checks)
            },
            "checks": [c.to_dict() for c in self.checks]
        }
        
        report_path = self.project_root / "logs" / "diagnostic_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Detailed report saved to: {report_path}")
        
        return passed, failed, warnings
    
    def apply_fixes(self):
        """Apply auto-fixes for fixable issues."""
        logger.info("Applying auto-fixes...")
        
        fixes_applied = 0
        
        for check in self.checks:
            if check.status in (CheckStatus.FAIL, CheckStatus.WARN) and check.fixable:
                if "Runtime Directory" in check.name:
                    dir_name = check.name.split(":")[1].strip()
                    dir_path = self.project_root / dir_name
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"  Created directory: {dir_name}/")
                    fixes_applied += 1
                
                elif check.name == "Kill Switch Status":
                    kill_switch_path = self.project_root / "runtime" / "KILL_SWITCH"
                    if kill_switch_path.exists():
                        kill_switch_path.unlink()
                        logger.info("  Removed KILL_SWITCH file")
                        fixes_applied += 1
        
        logger.info(f"Applied {fixes_applied} fixes")
        return fixes_applied


def main():
    parser = argparse.ArgumentParser(
        description="MiniQuantFund Trading System Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python diagnose_trading_execution.py          # Run diagnostics
    python diagnose_trading_execution.py --fix    # Run and auto-fix issues
    python diagnose_trading_execution.py --json   # Output JSON report
        """
    )
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues where possible")
    parser.add_argument("--json", action="store_true", help="Output JSON report to stdout")
    args = parser.parse_args()
    
    diagnostics = TradingSystemDiagnostics(auto_fix=args.fix)
    passed, failed, warnings = diagnostics.run_all_checks()
    
    if args.fix:
        diagnostics.apply_fixes()
        # Re-run to verify fixes
        passed, failed, warnings = diagnostics.run_all_checks()
    
    if args.json:
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "total": len(diagnostics.checks)
            },
            "checks": [c.to_dict() for c in diagnostics.checks]
        }
        print(json.dumps(report, indent=2))
    
    # Exit with appropriate code
    if failed > 0:
        sys.exit(1)
    elif warnings > 0:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

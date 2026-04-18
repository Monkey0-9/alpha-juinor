#!/usr/bin/env python3
"""
=============================================================================
NEXUS INSTITUTIONAL v0.3.0 - 24/7 CONTINUOUS EXECUTION MONITOR
=============================================================================
Runs the trading platform continuously with automatic recovery and logging.
Logs all activity to file for audit trail and debugging.
"""

import os
import sys
import logging
import json
import time
import subprocess
import socket
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import traceback

# Configure logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"nexus_24_7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
METRICS_FILE = LOG_DIR / f"metrics_24_7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Nexus24x7")

class Nexus24x7Monitor:
    """Manages continuous 24/7 execution of Nexus platform."""
    
    def __init__(self, mode: str = "backtest", asset_class: str = "multi", venues: int = 235):
        self.mode = mode
        self.asset_class = asset_class
        self.venues = venues
        self.process: Optional[subprocess.Popen] = None
        self.start_time = datetime.now()
        self.restart_count = 0
        self.error_count = 0
        self.metrics = {
            "start_time": self.start_time.isoformat(),
            "uptime_seconds": 0,
            "restarts": 0,
            "errors": 0,
            "execution_cycles": 0,
            "last_status": "INITIALIZING"
        }
        self.cycle_count = 0
        self.health_checks = []
        
    def log_metrics(self):
        """Save metrics to file."""
        self.metrics["uptime_seconds"] = int((datetime.now() - self.start_time).total_seconds())
        self.metrics["restarts"] = self.restart_count
        self.metrics["errors"] = self.error_count
        self.metrics["execution_cycles"] = self.cycle_count
        self.metrics["last_update"] = datetime.now().isoformat()
        
        with open(METRICS_FILE, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def health_check(self) -> bool:
        """Check if process is healthy."""
        if self.process is None:
            return False
        
        if self.process.poll() is not None:  # Process has terminated
            logger.warning(f"Process terminated with code: {self.process.poll()}")
            return False
        
        try:
            # Check if process is responsive
            psutil_available = False
            try:
                import psutil
                psutil_available = True
                process = psutil.Process(self.process.pid)
                if not process.is_running():
                    return False
            except:
                pass
            
            return True
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    def start_platform(self):
        """Start the Nexus institutional platform."""
        try:
            logger.info("="*80)
            logger.info(f"STARTING NEXUS INSTITUTIONAL PLATFORM - CYCLE {self.cycle_count + 1}")
            logger.info("="*80)
            logger.info(f"Mode: {self.mode}")
            logger.info(f"Asset Class: {self.asset_class}")
            logger.info(f"Venues: {self.venues}")
            logger.info(f"Configuration: config/production.yaml")
            
            # Use run_institutional_backtest.py which is more stable than nexus_institutional.py
            # This ensures the 24/7 monitor can run reliably
            if self.mode == "backtest":
                cmd = [
                    "python",
                    "run_institutional_backtest.py"
                ]
            else:
                cmd = [
                    "python",
                    "nexus_institutional.py",
                    "--mode", self.mode,
                    "--asset-class", self.asset_class,
                    "--venues", str(self.venues),
                    "--config", "config/production.yaml"
                ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            self.cycle_count += 1
            logger.info(f"Process started with PID: {self.process.pid}")
            self.metrics["last_status"] = "RUNNING"
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start platform: {e}")
            logger.error(traceback.format_exc())
            self.error_count += 1
            self.metrics["last_status"] = "ERROR"
            return False
    
    def handle_process_output(self):
        """Read and log process output."""
        try:
            if self.process and self.process.stdout:
                try:
                    # Try non-blocking read
                    import select
                    ready = select.select([self.process.stdout], [], [], 0.1)
                    if ready[0]:
                        line = self.process.stdout.readline()
                        if line:
                            logger.info(f"[PLATFORM] {line.rstrip()}")
                            return True
                except:
                    # Fallback for Windows (select doesn't work on pipes)
                    try:
                        line = self.process.stdout.readline()
                        if line:
                            logger.info(f"[PLATFORM] {line.rstrip()}")
                            return True
                    except:
                        pass
            return False
        except Exception as e:
            logger.debug(f"Output read error: {e}")
            return False
    
    def run_continuous(self, duration_seconds: Optional[int] = None):
        """Run platform continuously with monitoring."""
        logger.info("Starting 24/7 execution monitor...")
        logger.info(f"Log file: {LOG_FILE}")
        logger.info(f"Metrics file: {METRICS_FILE}")
        
        cycle_duration = 300  # 5 minutes per cycle for backtest mode
        if self.mode in ["paper", "live", "market-making"]:
            cycle_duration = 3600  # 1 hour for live modes
        
        last_metric_log = datetime.now()
        end_time = datetime.now() + timedelta(seconds=duration_seconds) if duration_seconds else None
        
        try:
            while True:
                # Check if we've exceeded the duration
                if end_time and datetime.now() > end_time:
                    logger.info(f"Duration limit reached: {duration_seconds} seconds")
                    break
                
                # Start or restart process if needed
                if self.process is None or not self.health_check():
                    if self.process is not None:
                        self.restart_count += 1
                        logger.warning(f"Restarting platform... (restart #{self.restart_count})")
                        try:
                            self.process.terminate()
                            self.process.wait(timeout=5)
                        except:
                            self.process.kill()
                    
                    if not self.start_platform():
                        logger.error("Failed to start platform, retrying in 30 seconds...")
                        time.sleep(30)
                        continue
                
                # Read and log output
                self.handle_process_output()
                
                # Log metrics periodically
                if (datetime.now() - last_metric_log).total_seconds() > 60:
                    self.log_metrics()
                    uptime = datetime.now() - self.start_time
                    logger.info(f"Uptime: {uptime}, Cycles: {self.cycle_count}, Restarts: {self.restart_count}")
                    last_metric_log = datetime.now()
                
                # Small sleep to prevent busy waiting
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, gracefully shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in continuous run: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of the platform."""
        logger.info("="*80)
        logger.info("SHUTTING DOWN NEXUS INSTITUTIONAL PLATFORM")
        logger.info("="*80)
        
        if self.process:
            try:
                logger.info("Terminating process...")
                self.process.terminate()
                self.process.wait(timeout=10)
                logger.info("Process terminated gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Process did not terminate, forcing kill...")
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
        
        self.metrics["last_status"] = "SHUTDOWN"
        self.metrics["total_runtime"] = int((datetime.now() - self.start_time).total_seconds())
        self.log_metrics()
        
        logger.info(f"Final Metrics:")
        logger.info(f"  Total Runtime: {self.metrics['total_runtime']} seconds")
        logger.info(f"  Execution Cycles: {self.cycle_count}")
        logger.info(f"  Restarts: {self.restart_count}")
        logger.info(f"  Errors: {self.error_count}")
        logger.info(f"Log file saved: {LOG_FILE}")
        logger.info(f"Metrics file saved: {METRICS_FILE}")


def main():
    """Main entry point for 24/7 continuous execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Nexus Institutional 24/7 Continuous Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live", "market-making"],
        default="backtest",
        help="Execution mode (default: backtest)"
    )
    
    parser.add_argument(
        "--asset-class",
        choices=["equities", "fixed-income", "crypto", "derivatives", "fx", "multi"],
        default="multi",
        help="Asset class (default: multi)"
    )
    
    parser.add_argument(
        "--venues",
        type=int,
        choices=[10, 50, 100, 235],
        default=235,
        help="Number of venues (default: 235)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duration in seconds (default: infinite)"
    )
    
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Number of execution cycles (default: infinite)"
    )
    
    args = parser.parse_args()
    
    monitor = Nexus24x7Monitor(
        mode=args.mode,
        asset_class=args.asset_class,
        venues=args.venues
    )
    
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*15 + "NEXUS INSTITUTIONAL 24/7 CONTINUOUS MONITOR" + " "*20 + "║")
    logger.info("║" + " "*20 + "Enterprise Trading Platform v0.3.0" + " "*24 + "║")
    logger.info("╚" + "="*78 + "╝")
    logger.info("")
    logger.info(f"Configuration:")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Asset Classes: {args.asset_class}")
    logger.info(f"  Venues: {args.venues}")
    logger.info(f"  Duration: {args.duration} seconds" if args.duration else "  Duration: Infinite")
    logger.info(f"  Cycles: {args.cycles}" if args.cycles else "  Cycles: Infinite")
    logger.info("")
    
    monitor.run_continuous(duration_seconds=args.duration)


if __name__ == "__main__":
    main()

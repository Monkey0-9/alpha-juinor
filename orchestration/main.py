"""
orchestration/main.py

Main entry point for the Mini Quant Fund orchestration system.
Provides industrial-grade deployment with comprehensive monitoring.
"""

import asyncio
import signal
import sys
import os
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from monitoring.structured_logger import get_logger
from orchestration.orchestrator import orchestrator, SystemMode
from infrastructure.infrastructure_guard import infrastructure_guard

# Try to import aiohttp for web server
try:
    from aiohttp import web
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    web = None

logger = get_logger("main")


class QuantFundSystem:
    """Main system controller for Mini Quant Fund."""

    def __init__(self):
        self.logger = logger
        self.orchestrator = orchestrator
        self.running = False
        self.shutdown_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.shutdown_requested = True

    async def start(self):
        """Start the trading system."""
        self.logger.info("Starting Mini Quant Fund System")

        try:
            # Determine system mode from environment
            execution_mode = os.getenv("EXECUTION_MODE", "paper").upper()

            if execution_mode == "LIVE":
                system_mode = SystemMode.LIVE_TRADING
            elif execution_mode == "PAPER":
                system_mode = SystemMode.PAPER_TRADING
            else:
                system_mode = SystemMode.SIMULATION

            self.orchestrator.set_system_mode(system_mode)

            # Start the orchestrator
            await self.orchestrator.start()
            self.running = True

            self.logger.info(
                f"Mini Quant Fund started successfully",
                mode=system_mode.value,
                timestamp=datetime.utcnow().isoformat()
            )

            # Main operational loop
            await self._operational_loop()

        except Exception as e:
            self.logger.critical(f"Failed to start system: {e}")
            raise
        finally:
            await self.stop()

    async def stop(self):
        """Stop the trading system."""
        if not self.running:
            return

        self.logger.info("Stopping Mini Quant Fund System")
        self.running = False

        try:
            await self.orchestrator.stop()
            self.logger.info("Mini Quant Fund stopped successfully")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    async def _operational_loop(self):
        """Main operational loop."""
        while self.running and not self.shutdown_requested:
            try:
                # Log system status periodically
                status = self.orchestrator.get_system_status()

                self.logger.info(
                    "System status update",
                    state=status["orchestrator"]["state"],
                    mode=status["orchestrator"]["mode"],
                    uptime_seconds=status["orchestrator"]["uptime_seconds"],
                    active_orders=len(status["execution"]),
                    infrastructure_health=status["infrastructure"]["health_status"]
                )

                # Check for shutdown conditions
                if self.shutdown_requested:
                    self.logger.info("Shutdown requested, exiting operational loop")
                    break

                # Sleep for next status update
                await asyncio.sleep(60)  # Update every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in operational loop: {e}")
                await asyncio.sleep(10)  # Brief pause before retry

    async def health_check_endpoint(self):
        """Health check endpoint for monitoring."""
        try:
            health = await infrastructure_guard.pre_flight_check()
            return {
                "status": "healthy" if health.status.value == "HEALTHY" else "degraded",
                "timestamp": datetime.utcnow().isoformat(),
                "system": self.orchestrator.get_system_status()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }


async def create_health_server():
    """Create simple health check server."""
    if not HAS_AIOHTTP or web is None:
        logger.warning("aiohttp not available, health server disabled")
        return None

    system = QuantFundSystem()

    async def health_handler(request):
        """Health check endpoint."""
        health_data = await system.health_check_endpoint()
        return web.json_response(health_data)

    async def status_handler(request):
        """Detailed status endpoint."""
        status = system.orchestrator.get_system_status()
        return web.json_response(status)

    async def metrics_handler(request):
        """Metrics endpoint for Prometheus."""
        # This would integrate with actual metrics collection
        metrics_text = """
# HELP quant_fund_uptime_seconds System uptime in seconds
# TYPE quant_fund_uptime_seconds counter
quant_fund_uptime_seconds 123456

# HELP quant_fund_active_orders Current number of active orders
# TYPE quant_fund_active_orders gauge
quant_fund_active_orders 0

# HELP quant_fund_trades_total Total number of executed trades
# TYPE quant_fund_trades_total counter
quant_fund_trades_total 0
        """.strip()

        return web.Response(
            text=metrics_text,
            content_type="text/plain; version=0.0.4; charset=utf-8"
        )

    app = web.Application()
    app.router.add_get('/health', health_handler)
    app.router.add_get('/status', status_handler)
    app.router.add_get('/metrics', metrics_handler)

    return app


async def main():
    """Main entry point."""
    logger.info("Mini Quant Fund - Industrial Grade Trading System")
    logger.info("Starting orchestration main entry point")

    runner = None

    try:
        # Create and start health server (if aiohttp available)
        app = await create_health_server()
        if app and HAS_AIOHTTP:
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', 8000)

            await site.start()
            logger.info("Health check server started on port 8000")
        else:
            logger.info("Running without health server (aiohttp not available)")

        # Start main system
        system = QuantFundSystem()
        await system.start()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if runner:
            await runner.cleanup()
        logger.info("Mini Quant Fund shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())

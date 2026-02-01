import os
import requests
import logging
from dataclasses import asdict

logger = logging.getLogger(__name__)


class AlpacaExecutionHandler:
    """
    Live execution handler using Alpaca Markets API.
    FREE paper trading with unlimited virtual capital.

    Setup same as AlpacaDataProvider.
    """

    def __init__(self, paper=True):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")

        if paper:
            self.base_url = os.getenv(
                "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
            )
        else:
            self.base_url = "https://api.alpaca.markets"

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }

        if not self.api_key:
            logger.warning(
                "Alpaca execution disabled: API keys not configured"
            )

        # Safety Integration
        try:
            from safety.circuit_breaker import CircuitBreaker, CircuitConfig
            import yaml

            # Load safety config
            safety_conf_path = "configs/safety_config.yaml"
            safety_cfg = {}
            if os.path.exists(safety_conf_path):
                try:
                    with open(safety_conf_path, "r") as f:
                        safety_cfg = yaml.safe_load(f) or {}
                except Exception as e:
                    logger.error(f"Failed to load safety config: {e}")

            # Extract circuit params
            circuit_params = safety_cfg.get("circuit_breaker", {})
            limits = circuit_params.get("limits", {})

            # Initialize with config or defaults
            cc = CircuitConfig(
                nav_usd=float(
                    circuit_params.get("nav_usd", 1_000_000.0)
                ),
                max_single_trade_loss_pct=float(
                    limits.get("max_single_trade_loss_pct", 0.01)
                ),
                max_daily_loss_pct=float(
                    limits.get("max_daily_loss_pct", 0.02)
                ),
                max_weekly_loss_pct=float(
                    limits.get("max_weekly_loss_pct", 0.05)
                )
            )
            self.circuit_breaker = CircuitBreaker(cc)
            logger.info(
                f"Circuit Breaker initialized with limits: {asdict(cc)}"
            )
        except ImportError:
            logger.warning("Safety module not found, circuit breaker disabled")
            self.circuit_breaker = None

    def submit_order(
        self, ticker: str, quantity: float, order_type: str = "market"
    ):
        """
        Submit order to Alpaca.
        quantity > 0: BUY
        quantity < 0: SELL
        """
        if not self.api_key:
            logger.error("Cannot submit order: Alpaca not configured")
            return None

        # Circuit Breaker Check
        if self.circuit_breaker and self.circuit_breaker.is_halted():
            logger.critical("Order REJECTED: Circuit breaker is HALTED")
            return None

        try:
            side = "buy" if quantity > 0 else "sell"
            qty = abs(quantity)

            # Alpaca requires integer shares for stocks
            qty = int(qty)
            if qty == 0:
                logger.warning(f"Order quantity rounded to zero for {ticker}")
                return None

            payload = {
                "symbol": ticker,
                "qty": qty,
                "side": side,
                "type": order_type,
                "time_in_force": "day"
            }

            url = f"{self.base_url}/v2/orders"
            response = requests.post(
                url, headers=self.headers, json=payload, timeout=10
            )
            response.raise_for_status()

            order_data = response.json()
            logger.info(
                f"Order submitted: {side} {qty} {ticker}, "
                f"ID: {order_data.get('id')}"
            )
            return order_data

        except Exception as e:
            logger.error(f"Failed to submit order for {ticker}: {e}")
            return None

    def get_account(self):
        """Get account details (cash, equity, buying power)."""
        try:
            url = f"{self.base_url}/v2/account"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return None

    def get_positions(self):
        """Get current positions."""
        try:
            url = f"{self.base_url}/v2/positions"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_position(self, symbol: str):
        """Get position for a single symbol."""
        try:
            url = f"{self.base_url}/v2/positions/{symbol}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            # Simple object wrapper to match expected 'pos.qty' usage
            from collections import namedtuple
            data = response.json()
            Position = namedtuple('Position', ['qty', 'symbol'])
            return Position(qty=float(data.get('qty', 0)), symbol=symbol)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None  # No position
            logger.error(f"Failed to get position for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}")
            return None

    def get_orders(self, status="all"):
        """Get orders (open, closed, all)."""
        try:
            url = f"{self.base_url}/v2/orders"
            params = {"status": status, "limit": 500}
            response = requests.get(
                url, headers=self.headers, params=params, timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    def cancel_all_orders(self):
        """Cancel all open orders."""
        try:
            url = f"{self.base_url}/v2/orders"
            response = requests.delete(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return False

    def record_fill(self, pnl_usd: float, notional_usd: float):
        """Record a fill result to the circuit breaker."""
        if hasattr(self, 'circuit_breaker'):
            res = self.circuit_breaker.record_trade_result(
                pnl_usd, notional_usd
            )
            if res.get("halt"):
                logger.critical(
                    f"CIRCUIT BREAKER HALT TRIGGERED: {res.get('reason')}"
                )


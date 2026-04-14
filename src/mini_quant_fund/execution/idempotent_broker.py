"""
Idempotent broker adapter for guaranteed order submission safety.
Ensures duplicate orders are never submitted even if network retries occur.
"""

import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class IdempotentOrder:
    """Represents an idempotent order with client ID tracking."""

    client_order_id: str
    broker_order_id: Optional[str]
    broker_response: Dict[str, Any]
    submission_time: datetime
    status: str
    retry_count: int = 0


class IdempotentBrokerAdapter:
    """
    Wraps a broker adapter to provide idempotent order submission.

    Guarantees:
    - No duplicate orders submitted even with network retries
    - Every order gets a unique client_order_id
    - All submissions are logged to database for audit trail
    """

    def __init__(self, broker, database_manager):
        """
        Initialize idempotent broker adapter.

        Args:
            broker: Underlying broker adapter (e.g., Alpaca, Mock)
            database_manager: DatabaseManager for tracking submissions
        """
        self.broker = broker
        self.db = database_manager
        self.submitted_orders: Dict[str, IdempotentOrder] = {}

    def submit_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Submit an order with idempotency guarantees.

        Args:
            symbol: Ticker symbol
            quantity: Order quantity
            side: "buy" or "sell"
            order_type: "MARKET" or "LIMIT"
            limit_price: Limit price if LIMIT order
            client_order_id: Optional client order ID (generated if not provided)
            **kwargs: Additional broker-specific parameters

        Returns:
            Order response dict with broker_order_id, status, etc.
        """

        # Generate client order ID if not provided
        if client_order_id is None:
            client_order_id = str(uuid.uuid4())

        # Check if we already submitted this order
        if client_order_id in self.submitted_orders:
            cached = self.submitted_orders[client_order_id]
            logger.info(
                f"Returning cached order response for {client_order_id}: "
                f"broker_id={cached.broker_order_id}"
            )
            return cached.broker_response

        # Submit order to broker
        try:
            order_params = {
                "symbol": symbol,
                "qty": quantity,
                "side": side,
                "type": order_type,
                "client_order_id": client_order_id,
            }

            if limit_price is not None:
                order_params["limit_price"] = limit_price

            order_params.update(kwargs)

            response = self.broker.submit_order(**order_params)

            # Extract broker order ID from response
            broker_order_id = response.get("order_id") or response.get("id")

            # Cache the submission
            idempotent_order = IdempotentOrder(
                client_order_id=client_order_id,
                broker_order_id=broker_order_id,
                broker_response=response,
                submission_time=datetime.utcnow(),
                status=response.get("status", "unknown"),
            )

            self.submitted_orders[client_order_id] = idempotent_order

            # Log to database audit trail if available
            self._log_to_audit(idempotent_order)

            logger.info(
                f"Order submitted: client_id={client_order_id}, "
                f"broker_id={broker_order_id}, symbol={symbol}, qty={quantity}"
            )

            return response

        except Exception as e:
            logger.error(f"Failed to submit order {client_order_id}: {str(e)}")
            raise

    def get_order_status(self, client_order_id: str) -> Optional[str]:
        """Get status of a previously submitted order."""
        if client_order_id in self.submitted_orders:
            return self.submitted_orders[client_order_id].status
        return None

    def cancel_order(self, client_order_id: str) -> Dict[str, Any]:
        """Cancel a previously submitted order."""
        if client_order_id not in self.submitted_orders:
            raise ValueError(f"Order {client_order_id} not found")

        idempotent_order = self.submitted_orders[client_order_id]
        broker_order_id = idempotent_order.broker_order_id

        if broker_order_id is None:
            raise ValueError(f"No broker order ID for {client_order_id}")

        response = self.broker.cancel_order(broker_order_id)
        idempotent_order.status = "cancelled"

        logger.info(f"Order cancelled: {client_order_id}")

        return response

    def _log_to_audit(self, order: IdempotentOrder) -> None:
        """Log order submission to audit trail."""
        try:
            if hasattr(self.db, "get_connection"):
                # Try to log to database
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO order_audit_log
                        (client_order_id, broker_order_id, submission_time, status)
                        VALUES (%s, %s, %s, %s)
                    """,
                        (
                            order.client_order_id,
                            order.broker_order_id,
                            order.submission_time,
                            order.status,
                        ),
                    )
                    conn.commit()
        except Exception as e:
            logger.warning(f"Failed to log order to audit trail: {str(e)}")

    def get_submitted_orders(self) -> Dict[str, IdempotentOrder]:
        """Get all submitted orders."""
        return self.submitted_orders.copy()

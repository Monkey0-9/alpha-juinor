
import ctypes
import logging
import os
import time
from typing import Optional, Dict, Any, Tuple

# Configure logging
logger = logging.getLogger("FPGA_SDK")

class FPGASDK:
    """
    Python bindings for interacting with the FPGA-based matching engine.
    Supports both hardware interaction (via ctypes) and mock mode for development.
    """
    
    def __init__(self, lib_path: Optional[str] = None, mock: bool = True):
        self.mock = mock
        self.lib = None
        
        if not mock:
            if lib_path is None:
                # Default library path for Linux systems
                lib_path = "/usr/local/lib/libfpga_driver.so"
            
            if os.path.exists(lib_path):
                try:
                    self.lib = ctypes.CDLL(lib_path)
                    self._setup_ctypes()
                    logger.info(f"Loaded FPGA driver from {lib_path}")
                except Exception as e:
                    logger.error(f"Failed to load FPGA driver: {e}")
                    self.mock = True
            else:
                logger.warning(f"FPGA driver not found at {lib_path}. Falling back to mock mode.")
                self.mock = True
        
        if self.mock:
            logger.info("FPGA SDK initialized in MOCK mode.")
            self._mock_data = {
                "orders": {},
                "book": {"bids": [], "asks": []},
                "matches": []
            }

    def _setup_ctypes(self):
        """Configure ctypes function prototypes for the C driver."""
        if self.lib:
            # submit_order(uint64_t id, uint32_t price, uint32_t qty, int side)
            self.lib.submit_order.argtypes = [ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
            self.lib.submit_order.restype = ctypes.c_int
            
            # poll_match(uint64_t *aggr_id, uint64_t *pass_id, uint32_t *price, uint32_t *qty)
            self.lib.poll_match.argtypes = [
                ctypes.POINTER(ctypes.c_uint64), 
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32)
            ]
            self.lib.poll_match.restype = ctypes.c_int

    def submit_order(self, order_id: int, price: int, qty: int, side: int) -> bool:
        """
        Submit an order to the FPGA matching engine.
        side: 0 for Buy, 1 for Sell
        """
        if self.mock:
            # Simple mock matching logic
            opposite_side = "asks" if side == 0 else "bids"
            own_side = "bids" if side == 0 else "asks"
            
            # Check for matches in mock mode
            book = self._mock_data["book"][opposite_side]
            if book:
                best_opp_price = book[0][0]
                if (side == 0 and price >= best_opp_price) or (side == 1 and price <= best_opp_price):
                    # Match occurred
                    match_qty = min(qty, book[0][1])
                    match_price = best_opp_price
                    passive_id = book[0][2]
                    
                    self._mock_data["matches"].append({
                        "aggressor_id": order_id,
                        "passive_id": passive_id,
                        "price": match_price,
                        "qty": match_qty,
                        "timestamp": time.time_ns()
                    })
                    
                    # Update book
                    if match_qty == book[0][1]:
                        book.pop(0)
                    else:
                        book[0] = (book[0][0], book[0][1] - match_qty, book[0][2])
                    
                    qty -= match_qty
            
            if qty > 0:
                # Add remainder to book
                self._mock_data["book"][own_side].append((price, qty, order_id))
                self._mock_data["book"][own_side].sort(key=lambda x: x[0], reverse=(side == 0))
            
            return True
        else:
            try:
                result = self.lib.submit_order(order_id, price, qty, side)
                return result == 0
            except Exception as e:
                logger.error(f"Error submitting order to FPGA: {e}")
                return False

    def poll_match(self) -> Optional[Dict[str, Any]]:
        """Poll for match events from the FPGA."""
        if self.mock:
            if self._mock_data["matches"]:
                return self._mock_data["matches"].pop(0)
            return None
        else:
            aggr_id = ctypes.c_uint64()
            pass_id = ctypes.c_uint64()
            price = ctypes.c_uint32()
            qty = ctypes.c_uint32()
            
            result = self.lib.poll_match(
                ctypes.byref(aggr_id),
                ctypes.byref(pass_id),
                ctypes.byref(price),
                ctypes.byref(qty)
            )
            
            if result == 1:  # Match found
                return {
                    "aggressor_id": aggr_id.value,
                    "passive_id": pass_id.value,
                    "price": price.value,
                    "qty": qty.value,
                    "timestamp": time.time_ns()
                }
            return None

    def dma_transfer(self, host_addr: int, card_addr: int, length: int, direction: int) -> bool:
        """
        Initiate a DMA transfer.
        direction: 0 for host->card, 1 for card->host
        """
        if self.mock:
            # Simulate DMA latency
            time.sleep(length / 1e9) 
            return True
        else:
            try:
                result = self.lib.dma_transfer(
                    ctypes.c_uint64(host_addr),
                    ctypes.c_uint32(card_addr),
                    ctypes.c_uint32(length),
                    ctypes.c_int(direction)
                )
                return result == 0
            except Exception as e:
                logger.error(f"Error during FPGA DMA transfer: {e}")
                return False

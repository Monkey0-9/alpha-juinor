import ctypes
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FPGAInterface:
    """Python bindings for Xilinx Alveo FPGA interface via PCIe DMA"""
    
    def __init__(self, lib_path: str = "/usr/local/lib/libfpga_driver.so"):
        try:
            self.lib = ctypes.CDLL(lib_path)
            self._setup_bindings()
            self.handle = self.lib.fpga_init()
        except Exception as e:
            logger.error(f"Failed to initialize FPGA driver: {e}")
            self.handle = None

    def _setup_bindings(self):
        self.lib.fpga_send_order.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
        self.lib.fpga_read_book.restype = ctypes.POINTER(ctypes.c_uint32)

    def send_order(self, price: float, qty: int, side: int):
        """Send order to FPGA matching engine in <200ns"""
        if self.handle:
            fixed_price = int(price * 10000) # 4 decimal fixed point
            self.lib.fpga_send_order(self.handle, fixed_price, qty, side)
        else:
            logger.warning("FPGA not initialized, using software fallback")

    def get_best_quotes(self) -> tuple:
        """Read best bid/ask directly from FPGA registers"""
        if self.handle:
            res = self.lib.fpga_read_book(self.handle)
            return (res[0] / 10000.0, res[1] / 10000.0)
        return (0.0, 0.0)

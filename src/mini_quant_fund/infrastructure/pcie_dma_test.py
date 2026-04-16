import time
import numpy as np
import logging
from mini_quant_fund.infrastructure.fpga_sdk import FPGASDK

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DMA_Test")

def run_dma_benchmark(sdk, sizes_mb=[1, 10, 100, 1000]):
    """
    Test DMA throughput for various buffer sizes.
    """
    results = []
    
    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        
        # Allocate host memory (simulated for mock)
        host_data = np.random.bytes(size_bytes)
        host_addr = 0x100000000  # Example host address
        card_addr = 0x00000000   # Example card address
        
        logger.info(f"Testing DMA transfer of {size_mb} MB...")
        
        # Host -> Card
        start_time = time.perf_counter()
        success = sdk.dma_transfer(host_addr, card_addr, size_bytes, direction=0)
        end_time = time.perf_counter()
        
        if success:
            h2c_time = end_time - start_time
            h2c_speed = size_mb / h2c_time if h2c_time > 0 else 0
            logger.info(f"Host -> Card: {h2c_speed:.2f} MB/s")
        else:
            logger.error("Host -> Card transfer failed")
            continue
            
        # Card -> Host
        start_time = time.perf_counter()
        success = sdk.dma_transfer(host_addr, card_addr, size_bytes, direction=1)
        end_time = time.perf_counter()
        
        if success:
            c2h_time = end_time - start_time
            c2h_speed = size_mb / c2h_time if c2h_time > 0 else 0
            logger.info(f"Card -> Host: {c2h_speed:.2f} MB/s")
        else:
            logger.error("Card -> Host transfer failed")
            continue
            
        results.append({
            "size_mb": size_mb,
            "h2c_speed": h2c_speed,
            "c2h_speed": c2h_speed
        })
        
    return results

if __name__ == "__main__":
    # Initialize SDK in mock mode for testing
    sdk = FPGASDK(mock=True)
    
    print("Starting PCIe Gen4 x16 DMA Throughput Test (Simulated)")
    print("-" * 50)
    
    results = run_dma_benchmark(sdk)
    
    print("\nSummary Results:")
    print(f"{'Size (MB)':<10} | {'H2C (MB/s)':<15} | {'C2H (MB/s)':<15}")
    print("-" * 50)
    for res in results:
        print(f"{res['size_mb']:<10} | {res['h2c_speed']:<15.2f} | {res['c2h_speed']:<15.2f}")

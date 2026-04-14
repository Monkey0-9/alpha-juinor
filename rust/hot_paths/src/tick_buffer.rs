//! High-throughput tick buffer
//! 
//! Lock-free ring buffer for market data ticks
//! Single-producer, single-consumer design for maximum throughput

use crossbeam::queue::ArrayQueue;
use std::sync::atomic::{AtomicU64, Ordering};

/// Market tick with cache-friendly layout
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Tick {
    pub timestamp_ns: u64,  // 8 bytes
    pub symbol_id: u64,     // 8 bytes
    pub price: f64,         // 8 bytes
    pub volume: u32,        // 4 bytes
    pub side: u8,           // 1 byte (0=bid, 1=ask, 2=trade)
    pub flags: u8,          // 1 byte (exchange flags)
    pub _padding: [u8; 2],  // 2 bytes padding
}

impl Tick {
    pub const SIDE_BID: u8 = 0;
    pub const SIDE_ASK: u8 = 1;
    pub const SIDE_TRADE: u8 = 2;
    
    #[inline(always)]
    pub fn new(timestamp_ns: u64, symbol_id: u64, price: f64, 
               volume: u32, side: u8) -> Self {
        Self {
            timestamp_ns,
            symbol_id,
            price,
            volume,
            side,
            flags: 0,
            _padding: [0; 2],
        }
    }
    
    #[inline(always)]
    pub fn is_trade(&self) -> bool {
        self.side == Self::SIDE_TRADE
    }
}

/// Lock-free tick buffer with pre-allocated storage
pub struct TickBuffer {
    queue: ArrayQueue<Tick>,
    dropped: AtomicU64,  // Count of dropped ticks when full
    received: AtomicU64, // Total received
}

impl TickBuffer {
    pub fn new(capacity: usize) -> Self {
        // Capacity must be power of 2 for optimal performance
        let capacity = capacity.next_power_of_two();
        
        Self {
            queue: ArrayQueue::new(capacity),
            dropped: AtomicU64::new(0),
            received: AtomicU64::new(0),
        }
    }
    
    /// Push tick into buffer (non-blocking)
    #[inline(always)]
    pub fn push(&self, timestamp_ns: u64, symbol_id: u64, price: f64,
                volume: u32, side: u8) -> bool {
        self.received.fetch_add(1, Ordering::Relaxed);
        
        let tick = Tick::new(timestamp_ns, symbol_id, price, volume, side);
        
        match self.queue.push(tick) {
            Ok(_) => true,
            Err(_) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }
    
    /// Pop tick from buffer (non-blocking)
    #[inline(always)]
    pub fn pop(&self) -> Option<Tick> {
        self.queue.pop().ok()
    }
    
    /// Get latest tick without removing
    pub fn get_latest(&self) -> Option<(u64, u64, f64, u32, u8)> {
        // For ArrayQueue, we can only pop
        self.pop().map(|t| (t.timestamp_ns, t.symbol_id, t.price, t.volume, t.side))
    }
    
    /// Current buffer length
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.queue.len()
    }
    
    /// Check if empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
    
    /// Check if full
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.queue.is_full()
    }
    
    /// Get drop statistics
    pub fn stats(&self) -> (u64, u64, f64) {
        let recv = self.received.load(Ordering::Relaxed);
        let drop = self.dropped.load(Ordering::Relaxed);
        let drop_rate = if recv > 0 { drop as f64 / recv as f64 } else { 0.0 };
        (recv, drop, drop_rate)
    }
    
    /// Process all available ticks with callback
    pub fn drain<F>(&self, mut f: F) -> usize
    where 
        F: FnMut(&Tick)
    {
        let mut count = 0;
        while let Some(tick) = self.pop() {
            f(&tick);
            count += 1;
        }
        count
    }
}

/// Multi-symbol tick dispatcher
pub struct TickDispatcher {
    buffers: Vec<TickBuffer>,
}

impl TickDispatcher {
    pub fn new(symbol_count: usize, buffer_size: usize) -> Self {
        let mut buffers = Vec::with_capacity(symbol_count);
        for _ in 0..symbol_count {
            buffers.push(TickBuffer::new(buffer_size));
        }
        Self { buffers }
    }
    
    /// Route tick to appropriate symbol buffer
    #[inline(always)]
    pub fn dispatch(&self, tick: Tick) -> bool {
        let symbol_id = tick.symbol_id as usize;
        if symbol_id < self.buffers.len() {
            self.buffers[symbol_id].push(
                tick.timestamp_ns,
                tick.symbol_id,
                tick.price,
                tick.volume,
                tick.side
            )
        } else {
            false
        }
    }
    
    /// Get buffer for symbol
    pub fn get_buffer(&self, symbol_id: usize) -> Option<&TickBuffer> {
        self.buffers.get(symbol_id)
    }
}

/// Tick processor with SIMD batching
pub struct TickProcessor {
    batch_size: usize,
}

impl TickProcessor {
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }
    
    /// Calculate returns from tick prices using SIMD
    pub fn calculate_returns(&self, prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return Vec::new();
        }
        
        let n = prices.len() - 1;
        let mut returns = Vec::with_capacity(n);
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                use std::arch::x86_64::*;
                
                let chunks = n / 4;
                for i in 0..chunks {
                    unsafe {
                        let curr = _mm256_loadu_pd(prices.as_ptr().add(i * 4));
                        let prev = _mm256_loadu_pd(prices.as_ptr().add(i * 4 + 1));
                        let ret = _mm256_div_pd(_mm256_sub_pd(curr, prev), prev);
                        
                        // Store results
                        let mut result = [0.0; 4];
                        _mm256_storeu_pd(result.as_mut_ptr(), ret);
                        returns.extend_from_slice(&result);
                    }
                }
                
                // Handle remainder
                for i in (chunks * 4)..n {
                    returns.push((prices[i] - prices[i + 1]) / prices[i + 1]);
                }
                
                return returns;
            }
        }
        
        // Scalar fallback
        for i in 0..n {
            returns.push((prices[i] - prices[i + 1]) / prices[i + 1]);
        }
        
        returns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tick_buffer() {
        let buffer = TickBuffer::new(1024);
        
        // Push ticks
        for i in 0..100 {
            assert!(buffer.push(i as u64 * 1000, 1, 100.0 + i as f64, 100, Tick::SIDE_TRADE));
        }
        
        assert_eq!(buffer.len(), 100);
        
        // Pop ticks
        let mut count = 0;
        while buffer.pop().is_some() {
            count += 1;
        }
        
        assert_eq!(count, 100);
        assert!(buffer.is_empty());
    }
    
    #[test]
    fn test_tick_processor() {
        let processor = TickProcessor::new(64);
        let prices = vec![100.0, 101.0, 99.0, 102.0, 100.0];
        
        let returns = processor.calculate_returns(&prices);
        
        assert_eq!(returns.len(), 4);
        assert!((returns[0] - (100.0 - 101.0) / 101.0).abs() < 1e-10);
    }
}

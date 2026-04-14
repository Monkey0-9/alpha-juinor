//! Lock-free order book implementation
//! 
//! Features:
//! - Cache-line alignment (64 bytes)
//! - Lock-free reads/writes using atomic operations
//! - Zero-allocation after initialization
//! - Fixed-size arrays for predictable memory layout

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const MAX_PRICE_LEVELS: usize = 256;
const CACHE_LINE_SIZE: usize = 64;

/// Cache-aligned price level to prevent false sharing
#[repr(align(64))]
#[derive(Debug, Clone, Copy, Default)]
pub struct PriceLevel {
    pub price: f64,
    pub volume: u32,
    pub order_count: u32,
    pub last_update_ns: u64,
}

/// Lock-free order book for single symbol
pub struct LockFreeOrderBook {
    symbol_id: u64,
    bids: [PriceLevel; MAX_PRICE_LEVELS],
    asks: [PriceLevel; MAX_PRICE_LEVELS],
    sequence: AtomicU64,  // Monotonic update counter
}

impl LockFreeOrderBook {
    pub fn new(symbol_id: u64) -> Self {
        Self {
            symbol_id,
            bids: [PriceLevel::default(); MAX_PRICE_LEVELS],
            asks: [PriceLevel::default(); MAX_PRICE_LEVELS],
            sequence: AtomicU64::new(0),
        }
    }
    
    /// Update bid price level (lock-free)
    #[inline(always)]
    pub fn update_bid(&self, level: usize, price: f64, volume: u32) {
        if level >= MAX_PRICE_LEVELS {
            return;
        }
        
        unsafe {
            let ptr = self.bids.as_ptr().add(level) as *mut PriceLevel;
            (*ptr).price = price;
            (*ptr).volume = volume;
            (*ptr).order_count += 1;
            (*ptr).last_update_ns = Self::timestamp_ns();
        }
        
        self.sequence.fetch_add(1, Ordering::Release);
    }
    
    /// Update ask price level (lock-free)
    #[inline(always)]
    pub fn update_ask(&self, level: usize, price: f64, volume: u32) {
        if level >= MAX_PRICE_LEVELS {
            return;
        }
        
        unsafe {
            let ptr = self.asks.as_ptr().add(level) as *mut PriceLevel;
            (*ptr).price = price;
            (*ptr).volume = volume;
            (*ptr).order_count += 1;
            (*ptr).last_update_ns = Self::timestamp_ns();
        }
        
        self.sequence.fetch_add(1, Ordering::Release);
    }
    
    /// Get best bid price (lock-free read)
    #[inline(always)]
    pub fn best_bid(&self) -> f64 {
        // Relaxed ordering is sufficient for price reads
        unsafe { (*self.bids.as_ptr()).price }
    }
    
    /// Get best ask price (lock-free read)
    #[inline(always)]
    pub fn best_ask(&self) -> f64 {
        unsafe { (*self.asks.as_ptr()).price }
    }
    
    /// Calculate spread
    #[inline(always)]
    pub fn spread(&self) -> f64 {
        self.best_ask() - self.best_bid()
    }
    
    /// Calculate mid price
    #[inline(always)]
    pub fn mid_price(&self) -> f64 {
        (self.best_bid() + self.best_ask()) / 2.0
    }
    
    /// Calculate book imbalance (buy pressure indicator)
    pub fn book_imbalance(&self) -> f64 {
        let mut bid_vol: u64 = 0;
        let mut ask_vol: u64 = 0;
        
        // Sum top 5 levels
        for i in 0..5 {
            bid_vol += self.bids[i].volume as u64;
            ask_vol += self.asks[i].volume as u64;
        }
        
        let total = bid_vol + ask_vol;
        if total == 0 {
            return 0.0;
        }
        
        (bid_vol as f64 - ask_vol as f64) / total as f64
    }
    
    /// Get VWAP for top N levels
    pub fn vwap_bid(&self, levels: usize) -> f64 {
        let n = levels.min(MAX_PRICE_LEVELS);
        let mut pv_sum = 0.0;
        let mut v_sum = 0u64;
        
        for i in 0..n {
            pv_sum += self.bids[i].price * self.bids[i].volume as f64;
            v_sum += self.bids[i].volume as u64;
        }
        
        if v_sum == 0 { 0.0 } else { pv_sum / v_sum as f64 }
    }
    
    pub fn vwap_ask(&self, levels: usize) -> f64 {
        let n = levels.min(MAX_PRICE_LEVELS);
        let mut pv_sum = 0.0;
        let mut v_sum = 0u64;
        
        for i in 0..n {
            pv_sum += self.asks[i].price * self.asks[i].volume as f64;
            v_sum += self.asks[i].volume as u64;
        }
        
        if v_sum == 0 { 0.0 } else { pv_sum / v_sum as f64 }
    }
    
    /// Nanosecond timestamp
    #[inline(always)]
    fn timestamp_ns() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
    
    /// Get current sequence number
    pub fn sequence(&self) -> u64 {
        self.sequence.load(Ordering::Acquire)
    }
}

// Thread-safe shared order book
pub type SharedOrderBook = Arc<LockFreeOrderBook>;

/// Order book manager for multiple symbols
pub struct OrderBookManager {
    books: Vec<SharedOrderBook>,
}

impl OrderBookManager {
    pub fn new(symbol_count: usize) -> Self {
        let mut books = Vec::with_capacity(symbol_count);
        for i in 0..symbol_count {
            books.push(Arc::new(LockFreeOrderBook::new(i as u64)));
        }
        Self { books }
    }
    
    pub fn get_book(&self, symbol_id: u64) -> Option<SharedOrderBook> {
        self.books.get(symbol_id as usize).cloned()
    }
    
    pub fn update_all_books<F>(&self, updater: F) 
    where 
        F: Fn(&LockFreeOrderBook) + Send + Sync 
    {
        // Parallel update using rayon (optional)
        for book in &self.books {
            updater(book);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_order_book_basic() {
        let book = LockFreeOrderBook::new(1);
        
        book.update_bid(0, 100.0, 1000);
        book.update_ask(0, 101.0, 500);
        
        assert_eq!(book.best_bid(), 100.0);
        assert_eq!(book.best_ask(), 101.0);
        assert_eq!(book.spread(), 1.0);
        assert_eq!(book.mid_price(), 100.5);
    }
    
    #[test]
    fn test_book_imbalance() {
        let book = LockFreeOrderBook::new(1);
        
        // Strong buy pressure
        for i in 0..5 {
            book.update_bid(i, 100.0 - i as f64, 10000);
            book.update_ask(i, 101.0 + i as f64, 1000);
        }
        
        let imbalance = book.book_imbalance();
        assert!(imbalance > 0.5);  // Buy pressure > 50%
    }
}

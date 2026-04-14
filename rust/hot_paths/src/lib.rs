use pyo3::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use crossbeam::queue::ArrayQueue;
use parking_lot::RwLock;

/// MiniQuantFund Rust Hot Paths
/// 
/// Safety-critical, ultra-low latency components:
/// - Lock-free ring buffers (crossbeam)
/// - Memory-safe order books
/// - SIMD-accelerated calculations
/// 
/// Target latency: < 1μs per operation

pub mod orderbook;
pub mod risk;
pub mod tick_buffer;

use orderbook::LockFreeOrderBook;
use tick_buffer::TickBuffer;

/// Python module initialization
#[pymodule]
fn mqf_hot_paths(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // OrderBook class
    m.add_class::<PyOrderBook>()?;
    
    // TickBuffer class  
    m.add_class::<PyTickBuffer>()?;
    
    // Utility functions
    m.add_wrapped(wrap_pyfunction!(get_timestamp_ns))?;
    m.add_wrapped(wrap_pyfunction!(measure_rdtsc_latency))?;
    m.add_wrapped(wrap_pyfunction!(simd_dot_product))?;
    
    Ok(())
}

/// Python wrapper for Rust OrderBook
#[pyclass]
pub struct PyOrderBook {
    inner: LockFreeOrderBook,
}

#[pymethods]
impl PyOrderBook {
    #[new]
    fn new(symbol_id: u64) -> Self {
        Self {
            inner: LockFreeOrderBook::new(symbol_id),
        }
    }
    
    fn update_bid(&self, level: usize, price: f64, volume: u32) {
        self.inner.update_bid(level, price, volume);
    }
    
    fn update_ask(&self, level: usize, price: f64, volume: u32) {
        self.inner.update_ask(level, price, volume);
    }
    
    fn get_best_bid(&self) -> f64 {
        self.inner.best_bid()
    }
    
    fn get_best_ask(&self) -> f64 {
        self.inner.best_ask()
    }
    
    fn get_spread(&self) -> f64 {
        self.inner.spread()
    }
    
    fn get_mid(&self) -> f64 {
        self.inner.mid_price()
    }
    
    fn get_book_imbalance(&self) -> f64 {
        self.inner.book_imbalance()
    }
}

/// Python wrapper for TickBuffer
#[pyclass]
pub struct PyTickBuffer {
    inner: TickBuffer,
}

#[pymethods]
impl PyTickBuffer {
    #[new]
    fn new(capacity: usize) -> Self {
        Self {
            inner: TickBuffer::new(capacity),
        }
    }
    
    fn push(&self, timestamp_ns: u64, symbol_id: u64, price: f64, 
            volume: u32, side: u8) -> bool {
        self.inner.push(timestamp_ns, symbol_id, price, volume, side)
    }
    
    fn len(&self) -> usize {
        self.inner.len()
    }
    
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    
    fn get_latest(&self) -> Option<(u64, u64, f64, u32, u8)> {
        self.inner.get_latest()
    }
}

/// Get nanosecond timestamp
#[pyfunction]
fn get_timestamp_ns() -> u64 {
    Instant::now()
        .elapsed()
        .as_nanos() as u64
}

/// Measure RDTSC-based latency in microseconds
#[pyfunction]
fn measure_rdtsc_latency() -> PyResult<f64> {
    let start = unsafe { core::arch::x86_64::_rdtsc() };
    
    // Do minimal work
    let mut x: u64 = 0;
    for i in 0..100 {
        x = x.wrapping_add(i);
    }
    std::hint::black_box(x);
    
    let end = unsafe { core::arch::x86_64::_rdtsc() };
    
    // Convert to microseconds (assuming 4GHz CPU)
    let cycles = end - start;
    let microseconds = cycles as f64 / 4000.0;
    
    Ok(microseconds)
}

/// SIMD-accelerated dot product using AVX2
#[pyfunction]
fn simd_dot_product(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Vectors must have same length"
        ));
    }
    
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        
        if is_x86_feature_detected!("avx2") {
            let mut sum = 0.0;
            let chunks = a.chunks_exact(4).zip(b.chunks_exact(4));
            
            for (chunk_a, chunk_b) in chunks {
                unsafe {
                    let va = _mm256_loadu_pd(chunk_a.as_ptr());
                    let vb = _mm256_loadu_pd(chunk_b.as_ptr());
                    let prod = _mm256_mul_pd(va, vb);
                    
                    // Horizontal sum
                    let hsum = _mm256_hadd_pd(prod, prod);
                    let sum_vec = _mm256_extractf128_pd(hsum, 0);
                    sum += _mm_cvtsd_f64(sum_vec);
                }
            }
            
            // Handle remainder
            let remainder = a.len() % 4;
            for i in (a.len() - remainder)..a.len() {
                sum += a[i] * b[i];
            }
            
            return Ok(sum);
        }
    }
    
    // Scalar fallback
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
}

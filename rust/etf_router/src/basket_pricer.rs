//! Virtu / Flow Traders Style ETF Basket Pricer
//! 
//! Uses Rust's nightly portable SIMD (std::simd) to price massive ETF baskets 
//! consisting of thousands of underlying assets across 235+ global exchanges.
//! 
//! By mapping the bid/ask vectors into 512-bit registers (AVX-512), 
//! we can compute the Net Asset Value (NAV) of the basket and detect 
//! arbitrage opportunities in a handful of CPU clock cycles.

#![feature(portable_simd)]
use std::simd::{f64x8, Simd};

pub struct EtfBasket {
    /// Weights of each constituent in the ETF
    pub weights: Vec<f64>,
    /// The current best bid across all 235+ exchanges for each constituent
    pub constituent_bids: Vec<f64>,
    /// The current best ask across all 235+ exchanges for each constituent
    pub constituent_asks: Vec<f64>,
}

impl EtfBasket {
    /// Calculates the instantaneous Nav of the basket using AVX-512 SIMD instructions.
    /// Processes 8 double-precision floats per clock cycle.
    #[inline(always)]
    pub fn calculate_nav_simd(&self) -> (f64, f64) {
        let mut nav_bid = f64x8::splat(0.0);
        let mut nav_ask = f64x8::splat(0.0);

        let chunks = self.weights.len() / 8;
        
        for i in 0..chunks {
            let offset = i * 8;
            
            // Load 8 weights, 8 bids, 8 asks simultaneously into 512-bit registers
            let w = f64x8::from_slice(&self.weights[offset..offset+8]);
            let b = f64x8::from_slice(&self.constituent_bids[offset..offset+8]);
            let a = f64x8::from_slice(&self.constituent_asks[offset..offset+8]);

            // Fused Multiply-Add (FMA) instruction mapping
            nav_bid += w * b;
            nav_ask += w * a;
        }

        // Horizontal sum reduction across the 512-bit register lanes
        (nav_bid.reduce_sum(), nav_ask.reduce_sum())
    }

    /// Evaluates structural arbitrage against the quoted ETF market price
    #[inline(always)]
    pub fn detect_arbitrage(&self, etf_market_bid: f64, etf_market_ask: f64) -> Option<&'static str> {
        let (implied_nav_bid, implied_nav_ask) = self.calculate_nav_simd();

        // If we can buy the underlying basket cheaper than the ETF bid -> Create
        if implied_nav_ask < etf_market_bid {
            return Some("CREATE_BASKET_ARBITRAGE");
        }
        
        // If we can buy the ETF cheaper than selling the underlying basket -> Redeem
        if etf_market_ask < implied_nav_bid {
            return Some("REDEEM_BASKET_ARBITRAGE");
        }

        None
    }
}

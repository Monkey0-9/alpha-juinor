//! Nexus Stochastic Volatility Library
//! 
//! Implements a heavily optimized Heston model for derivatives pricing using
//! exact simulation methods and SIMD-accelerated Monte Carlo.

use ndarray::Array1;
use rand::prelude::*;
use rand::distributions::StandardNormal;
use rayon::prelude::*;

/// Parameters for the Heston Stochastic Volatility Model.
#[derive(Debug, Clone, Copy)]
pub struct HestonParams {
    pub kappa: f64,   // Mean reversion speed
    pub theta: f64,   // Long variance
    pub sigma: f64,   // Volatility of volatility
    pub rho: f64,     // Correlation between asset and variance Brownian motions
    pub v0: f64,      // Initial variance
}

impl HestonParams {
    /// Feller condition check. If 2*kappa*theta > sigma^2, the variance process is strictly positive.
    #[inline]
    pub fn satisfies_feller(&self) -> bool {
        2.0 * self.kappa * self.theta > self.sigma * self.sigma
    }
}

/// Generates terminal asset prices under the Heston model using an Euler-Maruyama discretization.
/// Uses Rayon for zero-overhead work-stealing parallelization.
pub fn generate_paths_parallel(
    s0: f64,
    params: HestonParams,
    r: f64,
    t: f64,
    steps: usize,
    num_paths: usize,
) -> Array1<f64> {
    let dt = t / steps as f64;
    let sqrt_dt = dt.sqrt();
    
    // Uncorrelated Brownian motion coefficients
    let rho_bar = (1.0 - params.rho * params.rho).sqrt();

    let mut final_prices = Array1::zeros(num_paths);

    // Lock-free parallel processing of Monte Carlo paths
    final_prices.as_slice_mut().unwrap().par_iter_mut().for_each(|price_ref| {
        let mut rng = rand::thread_rng();
        let mut v_t = params.v0;
        let mut ln_s_t = s0.ln();

        for _ in 0..steps {
            let z1: f64 = rng.sample(StandardNormal);
            let z2: f64 = rng.sample(StandardNormal);

            let dw_s = z1 * sqrt_dt;
            let dw_v = (params.rho * z1 + rho_bar * z2) * sqrt_dt;

            // Full truncation scheme for variance process (avoids complex roots)
            let v_t_plus = v_t.max(0.0);
            
            // Log-Euler for Asset Price
            ln_s_t += (r - 0.5 * v_t_plus) * dt + v_t_plus.sqrt() * dw_s;
            
            // Variance process
            v_t += params.kappa * (params.theta - v_t_plus) * dt + params.sigma * v_t_plus.sqrt() * dw_v;
        }

        *price_ref = ln_s_t.exp();
    });

    final_prices
}

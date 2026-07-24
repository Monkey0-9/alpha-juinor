use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use std::env;

type float64 = f64;

#[derive(Debug, Deserialize)]
struct RiskInput {
    returns: Vec<float64>,
    confidence_level: float64,
}

#[derive(Debug, Serialize)]
struct RiskOutput {
    var: float64,
    cornish_fisher_var: float64,
    expected_shortfall: float64,
    spectral_risk_measure: float64,
    status: String,
}

fn calculate_var(returns: &[float64], confidence: float64) -> float64 {
    if returns.is_empty() { return 0.0; }
    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((1.0 - confidence) * sorted_returns.len() as float64).floor() as usize;
    sorted_returns[index.min(sorted_returns.len() - 1)]
}

fn calculate_cornish_fisher_var(returns: &[float64], confidence: float64) -> float64 {
    if returns.len() < 5 { return calculate_var(returns, confidence); }

    let n = returns.len() as float64;
    let mean = returns.iter().sum::<float64>() / n;
    let variance = returns.iter().map(|&r| (r - mean).powi(2)).sum::<float64>() / (n - 1.0);
    let std_dev = variance.sqrt();

    if std_dev < 1e-9 { return mean; }

    let skewness = (returns.iter().map(|&r| ((r - mean) / std_dev).powi(3)).sum::<float64>() * n) / ((n - 1.0) * (n - 2.0));
    let kurtosis = (returns.iter().map(|&r| ((r - mean) / std_dev).powi(4)).sum::<float64>() * n * (n + 1.0))
        / ((n - 1.0) * (n - 2.0) * (n - 3.0)) - 3.0; // Excess kurtosis

    // Gaussian inverse CDF approximation for 95% (z = -1.64485)
    let z = match confidence {
        c if c >= 0.99 => -2.32635,
        c if c >= 0.95 => -1.64485,
        _ => -1.28155,
    };

    let z_cf = z + (skewness / 6.0) * (z * z - 1.0)
        + (kurtosis / 24.0) * (z * z * z - 3.0 * z)
        - (skewness * skewness / 36.0) * (2.0 * z * z * z - 5.0 * z);

    mean + z_cf * std_dev
}

fn calculate_es(returns: &[float64], var: float64) -> float64 {
    let tail_returns: Vec<float64> = returns.par_iter()
        .filter(|&&r| r <= var)
        .cloned()
        .collect();
    
    if tail_returns.is_empty() {
        return var;
    }
    tail_returns.iter().sum::<float64>() / tail_returns.len() as float64
}

fn calculate_spectral_risk(returns: &[float64], gamma: float64) -> float64 {
    if returns.is_empty() { return 0.0; }
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let mut weighted_sum = 0.0;
    let mut weight_norm = 0.0;

    for (i, &r) in sorted.iter().enumerate() {
        let p = (i as float64 + 0.5) / n as float64;
        let weight = (-gamma * (1.0 - p)).exp(); // Exponential risk-aversion spectrum
        weighted_sum += weight * r;
        weight_norm += weight;
    }

    if weight_norm > 0.0 { weighted_sum / weight_norm } else { 0.0 }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        let output = RiskOutput {
            var: 0.0,
            cornish_fisher_var: 0.0,
            expected_shortfall: 0.0,
            spectral_risk_measure: 0.0,
            status: "Error: No input provided".to_string(),
        };
        println!("{}", serde_json::to_string(&output).unwrap());
        return;
    }

    let input: RiskInput = match serde_json::from_str(&args[1]) {
        Ok(data) => data,
        Err(_) => {
            let output = RiskOutput {
                var: 0.0,
                cornish_fisher_var: 0.0,
                expected_shortfall: 0.0,
                spectral_risk_measure: 0.0,
                status: "Error: Invalid JSON input".to_string(),
            };
            println!("{}", serde_json::to_string(&output).unwrap());
            return;
        }
    };

    let var = calculate_var(&input.returns, input.confidence_level);
    let cf_var = calculate_cornish_fisher_var(&input.returns, input.confidence_level);
    let es = calculate_es(&input.returns, var);
    let srm = calculate_spectral_risk(&input.returns, 10.0);

    let output = RiskOutput {
        var,
        cornish_fisher_var: cf_var,
        expected_shortfall: es,
        spectral_risk_measure: srm,
        status: "SUCCESS".to_string(),
    };

    println!("{}", serde_json::to_string(&output).unwrap());
}

use serde::{Deserialize, Serialize};
use serde_json::json;
use rayon::prelude::*;
use std::env;

#[derive(Debug, Deserialize)]
struct RiskInput {
    returns: Vec<float64>,
    confidence_level: float64,
}

#[derive(Debug, Serialize)]
struct RiskOutput {
    var: float64,
    expected_shortfall: float64,
    status: String,
}

type float64 = f64;

fn calculate_var(returns: &[float64], confidence: float64) -> float64 {
    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let index = ((1.0 - confidence) * sorted_returns.len() as float64).floor() as usize;
    sorted_returns[index]
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

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        let output = RiskOutput {
            var: 0.0,
            expected_shortfall: 0.0,
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
                expected_shortfall: 0.0,
                status: "Error: Invalid JSON input".to_string(),
            };
            println!("{}", serde_json::to_string(&output).unwrap());
            return;
        }
    };

    let var = calculate_var(&input.returns, input.confidence_level);
    let es = calculate_es(&input.returns, var);

    let output = RiskOutput {
        var,
        expected_shortfall: es,
        status: "SUCCESS".to_string(),
    };

    println!("{}", serde_json::to_string(&output).unwrap());
}

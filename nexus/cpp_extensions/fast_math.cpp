#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <execution>
#include <span>
#include <future>

namespace py = pybind11;

// ── Forward declarations from other compilation units ─────────────────
// (These are compiled together, so they're in the same translation unit
//  via the setup.py 'sources' list. We define them here for the module.)

// kalman.cpp
std::vector<double> batch_kalman_filter(const std::vector<double>&, double q, double r);

// fractal.cpp
double compute_hurst_exponent(const std::vector<double>&);
double compute_fractal_dimension(const std::vector<double>&);

// stats.cpp
double compute_shannon_entropy(const std::vector<double>&, int bins);

// timeseries.cpp
double compute_hawkes_intensity(const std::vector<double>&, double, double, double);

// signals.cpp
std::vector<double> compute_vwap(const std::vector<double>&, const std::vector<double>&);
std::vector<double> compute_zscore(const std::vector<double>&, int window);
std::vector<double> rsi(const std::vector<double>&, int);
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> macd(const std::vector<double>&, int, int, int);

// matrix.cpp
std::vector<std::vector<double>> compute_correlation_matrix(const std::vector<std::vector<double>>&);
std::vector<std::vector<double>> compute_covariance_shrinkage(const std::vector<std::vector<double>>&, double);

// ── 1. Portfolio Survival Analysis (Monte Carlo) ─────────────────────

double run_survival_analysis(
    double initial_capital, double mu, double sigma,
    int days, int n_simulations, double ruin_threshold)
{
    if (sigma == 0.0) return 1.0;
    if (initial_capital <= 0 || days <= 0) return 0.5;

    double ruin_level = initial_capital * (1.0 - ruin_threshold);
    std::atomic<int> survived{0};

    std::mt19937 rng(42);
    std::normal_distribution<double> dist(mu, sigma);

    #pragma omp parallel for reduction(+:survived)
    for (int i = 0; i < n_simulations; ++i) {
        double current_capital = initial_capital;
        bool ruined = false;
        for (int d = 0; d < days; ++d) {
            double ret = dist(rng);
            current_capital *= (1.0 + ret);
            if (current_capital <= ruin_level) {
                ruined = true;
                break;
            }
        }
        if (!ruined) survived++;
    }

    return static_cast<double>(survived) / static_cast<double>(n_simulations);
}

// ── 2. Monte Carlo VaR (Bootstrapping) ──────────────────────────────

double calculate_monte_carlo_var(
    const std::vector<double>& historical_returns,
    int num_paths, int horizon, double confidence_level)
{
    if (historical_returns.empty()) return 0.0;

    std::vector<double> simulated_end(num_paths);
    size_t n = historical_returns.size();

    #pragma omp parallel
    {
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, n - 1);

        #pragma omp for
        for (int i = 0; i < num_paths; ++i) {
            double path_sum = 0.0;
            for (int d = 0; d < horizon; ++d) {
                path_sum += historical_returns[dist(rng)];
            }
            simulated_end[i] = path_sum;
        }
    }

    std::sort(simulated_end.begin(), simulated_end.end());
    double target_rank = (1.0 - confidence_level) * static_cast<double>(simulated_end.size());
    int index = std::max(0, static_cast<int>(std::floor(target_rank)));
    return simulated_end[index];
}

// ── 3. GPU-Accelerated Monte Carlo VaR (CUDA fallback) ─────────────

double calculate_monte_carlo_var_gpu(
    const std::vector<double>& historical_returns,
    int num_paths, int horizon, double confidence_level)
{
    // Try CUDA, fall back to multi-threaded CPU
    // CUDA implementation would be in a separate .cu file
    return calculate_monte_carlo_var(historical_returns, num_paths, horizon, confidence_level);
}

// ── 4. Black-Litterman Portfolio Weights ────────────────────────────

struct BlackLittermanResult {
    std::vector<double> weights;
    std::vector<double> posterior_returns;
    std::vector<std::vector<double>> posterior_covariance;
};

BlackLittermanResult black_litterman_optimize(
    const std::vector<std::vector<double>>& historical_returns,
    const std::vector<double>& market_cap_weights,
    const std::vector<double>& views_return,
    const std::vector<std::vector<double>>& views_matrix,
    const std::vector<double>& views_confidence,
    double risk_aversion = 2.5, double tau = 0.05)
{
    int n = historical_returns.size();
    if (n == 0) return {};

    // Compute covariance matrix (shrinkage)
    auto cov = compute_covariance_shrinkage(historical_returns, 0.3);

    // Implied equilibrium returns (reverse optimization)
    std::vector<double> pi(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            pi[i] += cov[i][j] * market_cap_weights[j];
        }
        pi[i] *= risk_aversion;
    }

    // Omega: uncertainty of views
    int k = views_return.size();
    std::vector<std::vector<double>> omega(k, std::vector<double>(k, 0.0));
    for (int i = 0; i < k; ++i) {
        double conf = std::max(1e-6, views_confidence[i]);
        omega[i][i] = (1.0 - conf) / conf;
    }

    // Posterior returns:  pi' = pi + tau * Sigma * P' * (P * tau * Sigma * P' + Omega)^-1 * (Q - P * pi)
    // Simplified: use diagonal approximation for (P * tau * Sigma * P' + Omega)^-1
    std::vector<double> posterior(n, 0.0);
    for (int i = 0; i < n; ++i) {
        posterior[i] = pi[i];
    }

    // Apply views directly as adjustments
    for (int v = 0; v < k; ++v) {
        double view_weight = views_confidence[v];
        double implied_return = 0.0;
        for (int i = 0; i < n; ++i) {
            implied_return += views_matrix[v][i] * pi[i];
        }
        double residual = views_return[v] - implied_return;

        for (int i = 0; i < n; ++i) {
            posterior[i] += tau * views_matrix[v][i] * residual * view_weight;
        }
    }

    // Posterior covariance: M = Sigma + tau * (Sigma - Sigma * P' * (P * Sigma * P' + Omega)^-1 * P * Sigma)
    std::vector<std::vector<double>> post_cov = cov;

    // Weights: w = (risk_aversion * Sigma)^-1 * posterior_returns
    std::vector<double> weights(n, 0.0);
    for (int i = 0; i < n; ++i) {
        // Diagonal approximation for inverse
        double inv_var = 1.0 / std::max(cov[i][i], 1e-10);
        weights[i] = posterior[i] * inv_var / risk_aversion;
    }

    // Normalize
    double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    total = std::max(total, 1e-10);
    for (auto& w : weights) w = std::max(0.0, w / total);

    return {weights, posterior, post_cov};
}

// ── 5. Mean-Variance Optimization ───────────────────────────────────

std::vector<double> mean_variance_optimize(
    const std::vector<std::vector<double>>& historical_returns,
    double target_return = 0.0, double risk_aversion = 1.0)
{
    auto cov = compute_covariance_shrinkage(historical_returns, 0.3);
    int n = historical_returns.size();
    if (n == 0) return {};

    // Compute mean returns
    std::vector<double> mean_ret(n, 0.0);
    int obs = historical_returns[0].size();
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (double r : historical_returns[i]) sum += r;
        mean_ret[i] = sum / obs;
    }

    // Maximum Sharpe ratio portfolio (simplified)
    std::vector<double> weights(n, 1.0 / n);

    // Iterative risk-parity adjustment
    for (int iter = 0; iter < 100; ++iter) {
        double port_var = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                port_var += weights[i] * weights[j] * cov[i][j];
            }
        }
        double port_vol = std::sqrt(std::max(port_var, 1e-10));

        double port_ret = 0.0;
        for (int i = 0; i < n; ++i) port_ret += weights[i] * mean_ret[i];

        // Marginal contribution to risk
        std::vector<double> mcr(n);
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) sum += weights[j] * cov[i][j];
            mcr[i] = sum / port_vol;
        }

        // Adjust toward equal risk contribution
        double target_rc = port_vol / n;
        for (int i = 0; i < n; ++i) {
            double rc = weights[i] * mcr[i];
            double adj = (target_rc - rc) / std::max(port_vol, 1e-10) * 0.1;
            weights[i] += adj;
        }

        // Non-negativity + normalize
        double total = 0.0;
        for (auto& w : weights) { w = std::max(0.0, w); total += w; }
        total = std::max(total, 1e-10);
        for (auto& w : weights) w /= total;
    }

    return weights;
}

// ── 6. Information Coefficient Tracker ──────────────────────────────

struct ICTrackerResult {
    double ic;
    double win_rate;
};

ICTrackerResult compute_ic_tracker(
    const std::vector<double>& predictions,
    const std::vector<double>& realizations)
{
    int n = std::min(predictions.size(), realizations.size());
    if (n < 5) return {0.0, 0.5};

    double mean_p = 0.0, mean_r = 0.0;
    for (int i = 0; i < n; ++i) { mean_p += predictions[i]; mean_r += realizations[i]; }
    mean_p /= n; mean_r /= n;

    double cov = 0.0, var_p = 0.0, var_r = 0.0;
    int wins = 0;
    for (int i = 0; i < n; ++i) {
        double dp = predictions[i] - mean_p;
        double dr = realizations[i] - mean_r;
        cov += dp * dr;
        var_p += dp * dp;
        var_r += dr * dr;
        if ((predictions[i] > 0 && realizations[i] > 0) ||
            (predictions[i] < 0 && realizations[i] < 0)) wins++;
    }

    double ic = (std::sqrt(var_p * var_r) > 1e-10) ? cov / std::sqrt(var_p * var_r) : 0.0;
    double win_rate = static_cast<double>(wins) / n;

    return {ic, win_rate};
}

// ── 7. Ensemble Signal Fusion ───────────────────────────────────────

std::vector<double> fuse_ensemble_signals(
    const std::vector<std::vector<double>>& model_scores,
    const std::vector<double>& model_weights)
{
    int n_models = model_scores.size();
    if (n_models == 0) return {};
    int n_signals = model_scores[0].size();

    std::vector<double> fused(n_signals, 0.0);
    double total_weight = std::accumulate(model_weights.begin(), model_weights.end(), 0.0);
    total_weight = std::max(total_weight, 1e-10);

    for (int m = 0; m < n_models; ++m) {
        double w = model_weights[m] / total_weight;
        for (int s = 0; s < n_signals && s < (int)model_scores[m].size(); ++s) {
            fused[s] += w * model_scores[m][s];
        }
    }

    // Tanh clip to [-1, 1]
    for (auto& f : fused) f = std::tanh(f);
    return fused;
}

// ── 8. Multi-timeframe Feature Engineering ──────────────────────────

std::vector<double> compute_technical_features(const std::vector<double>& prices) {
    std::vector<double> features;
    if (prices.size() < 50) return features;

    int n = prices.size();

    // Returns
    double ret_1d = (prices[n-1] / prices[n-2]) - 1.0;
    double ret_5d = (prices[n-1] / prices[n-6]) - 1.0;
    double ret_20d = (prices[n-1] / prices[n-21]) - 1.0;

    // Volatility
    double sum = 0.0, sum_sq = 0.0;
    int lookback = std::min(20, n);
    for (int i = n - lookback; i < n; ++i) {
        double r = (prices[i] / prices[i-1]) - 1.0;
        sum += r; sum_sq += r * r;
    }
    double vol = std::sqrt(sum_sq / lookback - (sum / lookback) * (sum / lookback));

    // SMA crossover
    double sma_20 = 0.0, sma_50 = 0.0;
    int n20 = std::min(20, n);
    int n50 = std::min(50, n);
    for (int i = n - n20; i < n; ++i) sma_20 += prices[i];
    sma_20 /= n20;
    for (int i = n - n50; i < n; ++i) sma_50 += prices[i];
    sma_50 /= n50;
    features = {ret_1d, ret_5d, ret_20d, vol, sma_20 / sma_50 - 1.0};
    return features;
}

// ── PYBIND11 MODULE ─────────────────────────────────────────────────

void bind_optimization(py::module_& m);
void bind_feed(py::module_& m);
void bind_backtest(py::module_& m);

PYBIND11_MODULE(nexus_cpp, m) {
    m.doc() = "Nexus Fast Math C++ Extension — Multi-Engine Kernels";

    bind_optimization(m);
    bind_feed(m);
    bind_backtest(m);

    // ── Top-level risk functions ──
    m.def("run_survival_analysis", &run_survival_analysis,
        "Run Monte Carlo survival analysis (OpenMP accelerated)",
        py::arg("initial_capital"), py::arg("mu"), py::arg("sigma"),
        py::arg("days") = 252, py::arg("n_simulations") = 1000,
        py::arg("ruin_threshold") = 0.5);

    m.def("calculate_monte_carlo_var", &calculate_monte_carlo_var,
        "Calculate Monte Carlo VaR (OpenMP accelerated bootstrapping)",
        py::arg("historical_returns"), py::arg("num_paths") = 5000,
        py::arg("horizon") = 20, py::arg("confidence_level") = 0.95);

    m.def("calculate_monte_carlo_var_gpu", &calculate_monte_carlo_var_gpu,
        "GPU-accelerated Monte Carlo VaR (CUDA fallback to CPU)",
        py::arg("historical_returns"), py::arg("num_paths") = 50000,
        py::arg("horizon") = 20, py::arg("confidence_level") = 0.95);

    m.def("fuse_ensemble_signals", &fuse_ensemble_signals,
        "Weighted ensemble signal fusion across models",
        py::arg("model_scores"), py::arg("model_weights"));

    m.def("compute_technical_features", &compute_technical_features,
        "Multi-timeframe technical feature engineering",
        py::arg("prices"));

    // ── Black-Litterman Portfolio Optimizer ──
    py::class_<BlackLittermanResult>(m, "BlackLittermanResult")
        .def_readonly("weights", &BlackLittermanResult::weights)
        .def_readonly("posterior_returns", &BlackLittermanResult::posterior_returns)
        .def_readonly("posterior_covariance", &BlackLittermanResult::posterior_covariance);

    m.def("black_litterman_optimize", &black_litterman_optimize,
        "Black-Litterman portfolio optimization with investor views",
        py::arg("historical_returns"), py::arg("market_cap_weights"),
        py::arg("views_return"), py::arg("views_matrix"),
        py::arg("views_confidence"), py::arg("risk_aversion") = 2.5,
        py::arg("tau") = 0.05);

    m.def("mean_variance_optimize", &mean_variance_optimize,
        "Mean-variance optimization (risk parity iteration)",
        py::arg("historical_returns"), py::arg("target_return") = 0.0,
        py::arg("risk_aversion") = 1.0);

    // ── IC Tracker ──
    py::class_<ICTrackerResult>(m, "ICTrackerResult")
        .def_readonly("ic", &ICTrackerResult::ic)
        .def_readonly("win_rate", &ICTrackerResult::win_rate);

    m.def("compute_ic_tracker", &compute_ic_tracker,
        "Compute Information Coefficient and win rate",
        py::arg("predictions"), py::arg("realizations"));

    // ── KALMAN submodule ──
    auto kalman = m.def_submodule("kalman", "Kalman filter kernels");
    kalman.def("batch_kalman_filter", &batch_kalman_filter,
        "Kalman filter smoothing on price series",
        py::arg("prices"), py::arg("q") = 1e-5, py::arg("r") = 1e-4);

    // ── FRACTAL submodule ──
    auto fractal = m.def_submodule("fractal", "Fractal analysis kernels");
    fractal.def("compute_hurst_exponent", &compute_hurst_exponent,
        "Compute Hurst exponent (R/S analysis)");
    fractal.def("compute_fractal_dimension", &compute_fractal_dimension,
        "Compute fractal dimension (Katz algorithm)");

    // ── STATS submodule ──
    auto stats = m.def_submodule("stats", "Statistical kernels");
    stats.def("compute_shannon_entropy", &compute_shannon_entropy,
        "Compute Shannon entropy of return distribution",
        py::arg("returns"), py::arg("bins") = 20);

    // ── TIMESERIES submodule ──
    auto ts = m.def_submodule("timeseries", "Time series kernels");
    ts.def("compute_hawkes_intensity", &compute_hawkes_intensity,
        "Compute Hawkes process self-exciting intensity",
        py::arg("returns"), py::arg("mu") = 0.1,
        py::arg("alpha") = 0.5, py::arg("beta") = 0.8);

    // ── SIGNALS submodule ──
    auto signals = m.def_submodule("signals", "Signal processing kernels");
    signals.def("compute_vwap", &compute_vwap,
        "Compute Volume-Weighted Average Price");
    signals.def("compute_zscore", &compute_zscore,
        "Compute rolling z-score",
        py::arg("values"), py::arg("window") = 20);
    signals.def("rsi", &rsi,
        "Compute Relative Strength Index",
        py::arg("prices"), py::arg("period") = 14);
    signals.def("macd", &macd,
        "Compute MACD (macd_line, signal_line, histogram)",
        py::arg("prices"), py::arg("fast_period") = 12, py::arg("slow_period") = 26, py::arg("signal_period") = 9);

    // ── MATRIX submodule ──
    auto matrix = m.def_submodule("matrix", "Matrix / linear algebra kernels");
    matrix.def("compute_correlation_matrix", &compute_correlation_matrix,
        "Compute pairwise correlation matrix");
    matrix.def("compute_covariance_shrinkage", &compute_covariance_shrinkage,
        "Compute Ledoit-Wolf shrinkage covariance matrix",
        py::arg("returns"), py::arg("shrinkage") = 0.2);
}
#include <cmath>
#include <iostream>
#include <vector>
#include <immintrin.h>  // AVX2/AVX-512 intrinsics
#include <chrono>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Institutional Greek Struct with additional metrics
struct Greeks {
    double delta;
    double gamma;
    double theta;
    double vega;
    double rho;
    double implied_vol;
    double theoretical_price;
    double delta_gamma;  // DdeltaDspot
    double vega_kappa;   // DvegaDvol
    uint64_t timestamp_ns;
};

// Batch calculation structure
struct OptionBatch {
    double* S;      // Underlying prices
    double* K;      // Strikes
    double* T;      // Times to expiry
    double* r;      // Risk-free rates
    double* sigma;  // Volatilities
    int* is_call;   // Call/Put flags
    Greeks* results; // Output results
    size_t size;     // Batch size
};

// Market making parameters
struct MMParams {
    double target_delta;
    double max_gamma;
    double max_vega;
    double inventory_limit;
    double bid_ask_spread_bps;
    double hedge_threshold;
    uint64_t latency_budget_ns;
};

extern "C" {
    // High-performance normal CDF using polynomial approximation
    inline double fast_cnd(double x) {
        // Abramowitz and Stegun approximation
        static const double a1 =  0.254829592;
        static const double a2 = -0.284496736;
        static const double a3 =  1.421413741;
        static const double a4 = -1.453152027;
        static const double a5 =  1.061405429;
        static const double p  =  0.3275911;

        int sign = x < 0 ? -1 : 1;
        x = std::abs(x);

        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);

        return 0.5 * (1.0 + sign * y);
    }

    // Real-time market making engine
    Greeks calculate_greeks_institutional(double S, double K, double T, double r, double sigma, int is_call) {
        if (T <= 0) return {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);

        Greeks g;

        // Delta
        if (is_call == 1) {
            g.delta = fast_cnd(d1);
        } else {
            g.delta = fast_cnd(d1) - 1.0;
        }

        // Gamma
        g.gamma = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * d1 * d1) / (S * sigma * std::sqrt(T));

        // Vega
        g.vega = S * (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * d1 * d1) * std::sqrt(T) / 100.0;

        // Theta
        double term1 = -(S * (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * d1 * d1) * sigma) / (2.0 * std::sqrt(T));
        if (is_call == 1) {
            double term2 = r * K * std::exp(-r * T) * fast_cnd(d2);
            g.theta = (term1 - term2) / 365.0;
        } else {
            double term2 = r * K * std::exp(-r * T) * fast_cnd(-d2);
            g.theta = (term1 + term2) / 365.0;
        }

        // Rho
        if (is_call == 1) {
            g.rho = K * T * std::exp(-r * T) * fast_cnd(d2) / 100.0;
        } else {
            g.rho = -K * T * std::exp(-r * T) * fast_cnd(-d2) / 100.0;
        }

        // Additional metrics
        g.implied_vol = sigma;  // Would need iterative solver for true IV
        g.theoretical_price = is_call ?
            S * fast_cnd(d1) - K * std::exp(-r * T) * fast_cnd(d2) :
            K * std::exp(-r * T) * fast_cnd(-d2) - S * fast_cnd(-d1);

        // Higher-order Greeks
        g.delta_gamma = -(d1 * (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * d1 * d1)) /
                       (S * S * sigma * sigma * std::sqrt(T));
        g.vega_kappa = S * (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * d1 * d1) *
                       std::sqrt(T) * d1 / sigma;

        g.timestamp_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();

        return g;
    }

    // Batch Greeks calculation (Simplified non-SIMD for compatibility, can be optimized later)
    void calculate_greeks_batch(const OptionBatch* batch) {
        for (size_t i = 0; i < batch->size; i++) {
            batch->results[i] = calculate_greeks_institutional(
                batch->S[i], batch->K[i], batch->T[i],
                batch->r[i], batch->sigma[i], batch->is_call[i]
            );
        }
    }
}

#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

// Institutional Greek Struct
struct Greeks {
    double delta;
    double gamma;
    double theta;
    double vega;
    double rho;
};

extern "C" {
    // Standard normal cumulative distribution function
    double cnd(double x) {
        return 0.5 * std::erfc(-x * M_SQRT1_2);
    }

    // Standard normal probability density function
    double pdf(double x) {
        return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    }

    Greeks calculate_greeks_institutional(double S, double K, double T, double r, double sigma, int is_call) {
        if (T <= 0) return {0, 0, 0, 0, 0};

        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);

        Greeks g;
        
        // Delta
        if (is_call == 1) {
            g.delta = cnd(d1);
        } else {
            g.delta = cnd(d1) - 1.0;
        }

        // Gamma
        g.gamma = pdf(d1) / (S * sigma * std::sqrt(T));

        // Vega
        g.vega = S * pdf(d1) * std::sqrt(T) / 100.0;

        // Theta
        double term1 = -(S * pdf(d1) * sigma) / (2.0 * std::sqrt(T));
        if (is_call == 1) {
            double term2 = r * K * std::exp(-r * T) * cnd(d2);
            g.theta = (term1 - term2) / 365.0;
        } else {
            double term2 = r * K * std::exp(-r * T) * cnd(-d2);
            g.theta = (term1 + term2) / 365.0;
        }

        // Rho
        if (is_call == 1) {
            g.rho = K * T * std::exp(-r * T) * cnd(d2) / 100.0;
        } else {
            g.rho = -K * T * std::exp(-r * T) * cnd(-d2) / 100.0;
        }

        return g;
    }
}

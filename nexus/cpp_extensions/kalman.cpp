#include <vector>

std::vector<double> batch_kalman_filter(const std::vector<double>& prices, double q = 1e-5, double r = 1e-4) {
    std::vector<double> filtered;
    if (prices.empty()) return filtered;
    filtered.reserve(prices.size());
    
    double x = prices[0]; // Initial state estimate
    double p = 1.0;       // Initial estimate uncertainty
    
    for (double z : prices) {
        // Predict step
        p = p + q;
        
        // Update step
        double k = p / (p + r);
        x = x + k * (z - x);
        p = (1.0 - k) * p;
        
        filtered.push_back(x);
    }
    return filtered;
}

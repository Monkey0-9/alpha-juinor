#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

double compute_hurst_exponent(const std::vector<double>& prices) {
    if(prices.size() < 10) return 0.5;
    
    std::vector<double> returns;
    returns.reserve(prices.size() - 1);
    for(size_t i = 1; i < prices.size(); ++i) {
        if(prices[i-1] != 0) {
            returns.push_back(std::log(prices[i] / prices[i-1]));
        } else {
            returns.push_back(0.0);
        }
    }
    
    int n = returns.size();
    if(n == 0) return 0.5;
    
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / n;
    
    std::vector<double> dev(n);
    double sq_sum = 0.0;
    for(int i = 0; i < n; ++i) {
        double d = returns[i] - mean;
        dev[i] = d;
        sq_sum += d * d;
    }
    double stdev = std::sqrt(sq_sum / n);
    if(stdev == 0) return 0.5;
    
    double max_cum = 0.0;
    double min_cum = 0.0;
    double cum = 0.0;
    for(int i = 0; i < n; ++i) {
        cum += dev[i];
        if(cum > max_cum) max_cum = cum;
        if(cum < min_cum) min_cum = cum;
    }
    
    double R = max_cum - min_cum;
    double RS = R / stdev;
    
    if(RS <= 0) return 0.5;
    return std::log(RS) / std::log(n);
}

double compute_fractal_dimension(const std::vector<double>& prices) {
    if(prices.size() < 10) return 1.5;
    
    int n = prices.size();
    double length = 0.0;
    for(int i = 1; i < n; ++i) {
        length += std::abs(prices[i] - prices[i-1]);
    }
    
    double min_p = *std::min_element(prices.begin(), prices.end());
    double max_p = *std::max_element(prices.begin(), prices.end());
    double range = max_p - min_p;
    
    if (range <= 0) return 1.0;
    
    return 1.0 + (std::log(length / range) / std::log(n));
}

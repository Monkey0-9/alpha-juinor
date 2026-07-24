#include <vector>
#include <cmath>

// Computes the Hawkes process intensity for a sequence of returns
// assuming discrete time steps where an 'event' is an extreme return.
double compute_hawkes_intensity(const std::vector<double>& returns, double mu = 0.1, double alpha = 0.5, double beta = 0.8) {
    if(returns.empty()) return mu;
    
    double decay = std::exp(-beta);
    double excitation = 0.0;
    
    // Calculate simple standard deviation to use as threshold
    double sum = 0.0;
    for(double r : returns) sum += r;
    double mean = sum / returns.size();
    
    double sq_sum = 0.0;
    for(double r : returns) sq_sum += (r - mean) * (r - mean);
    double stdev = std::sqrt(sq_sum / returns.size());
    
    if (stdev == 0) return mu;
    
    for(size_t i = 0; i < returns.size(); ++i) {
        excitation *= decay;
        if(std::abs(returns[i]) > stdev) { // Event: return > 1 stdev
            excitation += alpha;
        }
    }
    
    return mu + excitation;
}

#include <vector>
#include <cmath>
#include <algorithm>

double compute_shannon_entropy(const std::vector<double>& returns, int bins = 20) {
    if(returns.empty()) return 0.0;
    
    double min_val = *std::min_element(returns.begin(), returns.end());
    double max_val = *std::max_element(returns.begin(), returns.end());
    
    if(min_val == max_val) return 0.0;
    
    std::vector<int> counts(bins, 0);
    double bin_width = (max_val - min_val) / bins;
    
    for(double r : returns) {
        int bin = static_cast<int>((r - min_val) / bin_width);
        if(bin >= bins) bin = bins - 1;
        if(bin < 0) bin = 0;
        counts[bin]++;
    }
    
    double entropy = 0.0;
    double n = returns.size();
    for(int count : counts) {
        if(count > 0) {
            double p = count / n;
            entropy -= p * std::log(p);
        }
    }
    return entropy;
}

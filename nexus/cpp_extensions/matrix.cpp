#include <vector>
#include <cmath>

std::vector<std::vector<double>> compute_correlation_matrix(const std::vector<std::vector<double>>& returns) {
    int n_assets = returns.size();
    if(n_assets == 0) return {};
    int n_obs = returns[0].size();
    
    std::vector<std::vector<double>> corr(n_assets, std::vector<double>(n_assets, 1.0));
    
    std::vector<double> means(n_assets, 0.0);
    std::vector<double> stdevs(n_assets, 0.0);
    
    for(int i = 0; i < n_assets; ++i) {
        double sum = 0.0;
        for(double r : returns[i]) sum += r;
        means[i] = sum / n_obs;
        
        double sq_sum = 0.0;
        for(double r : returns[i]) sq_sum += (r - means[i]) * (r - means[i]);
        stdevs[i] = std::sqrt(sq_sum / n_obs);
    }
    
    for(int i = 0; i < n_assets; ++i) {
        for(int j = i + 1; j < n_assets; ++j) {
            double cov = 0.0;
            for(int k = 0; k < n_obs; ++k) {
                cov += (returns[i][k] - means[i]) * (returns[j][k] - means[j]);
            }
            cov /= n_obs;
            
            double denom = stdevs[i] * stdevs[j];
            double c = (denom == 0.0) ? 0.0 : cov / denom;
            corr[i][j] = c;
            corr[j][i] = c;
        }
    }
    return corr;
}

// Basic Ledoit-Wolf style linear shrinkage towards identity matrix
std::vector<std::vector<double>> compute_covariance_shrinkage(const std::vector<std::vector<double>>& returns, double shrinkage=0.2) {
    int n_assets = returns.size();
    if(n_assets == 0) return {};
    int n_obs = returns[0].size();
    
    std::vector<std::vector<double>> cov(n_assets, std::vector<double>(n_assets, 0.0));
    std::vector<double> means(n_assets, 0.0);
    
    for(int i = 0; i < n_assets; ++i) {
        double sum = 0.0;
        for(double r : returns[i]) sum += r;
        means[i] = sum / n_obs;
    }
    
    double trace = 0.0;
    for(int i = 0; i < n_assets; ++i) {
        for(int j = i; j < n_assets; ++j) {
            double c = 0.0;
            for(int k = 0; k < n_obs; ++k) {
                c += (returns[i][k] - means[i]) * (returns[j][k] - means[j]);
            }
            c /= n_obs;
            cov[i][j] = c;
            cov[j][i] = c;
            if(i == j) trace += c;
        }
    }
    
    // Target matrix: diagonal with average variance
    double avg_var = trace / n_assets;
    for(int i = 0; i < n_assets; ++i) {
        for(int j = 0; j < n_assets; ++j) {
            double target = (i == j) ? avg_var : 0.0;
            cov[i][j] = shrinkage * target + (1.0 - shrinkage) * cov[i][j];
        }
    }
    
    return cov;
}

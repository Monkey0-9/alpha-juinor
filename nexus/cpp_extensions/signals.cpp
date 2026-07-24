#include <vector>
#include <cmath>
#include <tuple>

std::vector<double> compute_vwap(const std::vector<double>& prices, const std::vector<double>& volumes) {
    std::vector<double> vwap;
    if(prices.empty() || volumes.empty() || prices.size() != volumes.size()) return vwap;
    
    vwap.reserve(prices.size());
    
    double cum_pv = 0.0;
    double cum_v = 0.0;
    
    for(size_t i = 0; i < prices.size(); ++i) {
        cum_pv += prices[i] * volumes[i];
        cum_v += volumes[i];
        if (cum_v == 0) {
            vwap.push_back(prices[i]);
        } else {
            vwap.push_back(cum_pv / cum_v);
        }
    }
    
    return vwap;
}

std::vector<double> compute_zscore(const std::vector<double>& values, int window) {
    std::vector<double> zscores;
    if(values.empty()) return zscores;
    
    zscores.reserve(values.size());
    
    for(size_t i = 0; i < values.size(); ++i) {
        if (i < (size_t)window - 1) {
            zscores.push_back(0.0);
            continue;
        }
        
        double sum = 0.0;
        for(size_t j = i - window + 1; j <= i; ++j) {
            sum += values[j];
        }
        double mean = sum / window;
        
        double sq_sum = 0.0;
        for(size_t j = i - window + 1; j <= i; ++j) {
            sq_sum += (values[j] - mean) * (values[j] - mean);
        }
        double stdev = std::sqrt(sq_sum / window);
        
        if (stdev == 0) {
            zscores.push_back(0.0);
        } else {
            zscores.push_back((values[i] - mean) / stdev);
        }
    }
    
    return zscores;
}

std::vector<double> rsi(const std::vector<double>& prices, int period) {
    std::vector<double> results;
    if (prices.size() <= (size_t)period) return results;
    
    double gain = 0.0, loss = 0.0;
    for (int i = 1; i <= period; ++i) {
        double diff = prices[i] - prices[i-1];
        if (diff > 0) gain += diff;
        else loss -= diff;
    }
    gain /= period;
    loss /= period;
    
    for (size_t i = period; i < prices.size(); ++i) {
        if (i > (size_t)period) {
            double diff = prices[i] - prices[i-1];
            double cur_gain = diff > 0 ? diff : 0;
            double cur_loss = diff < 0 ? -diff : 0;
            gain = (gain * (period - 1) + cur_gain) / period;
            loss = (loss * (period - 1) + cur_loss) / period;
        }
        if (loss == 0) results.push_back(100.0);
        else {
            double rs = gain / loss;
            results.push_back(100.0 - (100.0 / (1.0 + rs)));
        }
    }
    return results;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> macd(const std::vector<double>& prices, int fast_period, int slow_period, int signal_period) {
    auto ema = [](const std::vector<double>& data, int period) {
        std::vector<double> result;
        if (data.empty()) return result;
        double multiplier = 2.0 / (period + 1.0);
        double val = data[0];
        for (double d : data) {
            val = (d - val) * multiplier + val;
            result.push_back(val);
        }
        return result;
    };

    std::vector<double> fast_ema = ema(prices, fast_period);
    std::vector<double> slow_ema = ema(prices, slow_period);
    
    std::vector<double> macd_line;
    for (size_t i = 0; i < prices.size(); ++i) {
        macd_line.push_back(fast_ema[i] - slow_ema[i]);
    }
    
    std::vector<double> signal_line = ema(macd_line, signal_period);
    
    std::vector<double> histogram;
    for (size_t i = 0; i < prices.size(); ++i) {
        histogram.push_back(macd_line[i] - signal_line[i]);
    }
    
    return std::make_tuple(macd_line, signal_line, histogram);
}

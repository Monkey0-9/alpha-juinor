#include <vector>
#include <string>
#include <unordered_map>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class CppPortfolioOptimizer {
private:
    double max_position_size;
    double max_leverage;
    double min_weight;
    
public:
    CppPortfolioOptimizer(double max_pos = 0.2, double max_lev = 1.0, double min_w = 0.01)
        : max_position_size(max_pos), max_leverage(max_lev), min_weight(min_w) {}

    std::unordered_map<std::string, double> optimize_weights(
        const std::vector<std::string>& symbols,
        const std::vector<double>& signals,
        const std::vector<std::vector<double>>& correlation_matrix) 
    {
        if (symbols.size() != signals.size() || symbols.size() != correlation_matrix.size()) {
            throw std::invalid_argument("Size mismatch in optimize_weights");
        }

        int n = symbols.size();
        std::unordered_map<std::string, double> weights;
        
        if (n == 0) return weights;

        // 1. Initial weights based on signal magnitude (Softmax / L1 norm)
        std::vector<double> initial_w(n, 0.0);
        double sum_abs_signal = 0.0;
        for (double s : signals) sum_abs_signal += std::abs(s);
        
        if (sum_abs_signal > 0.0) {
            for (int i = 0; i < n; ++i) {
                initial_w[i] = signals[i] / sum_abs_signal;
            }
        }

        // 2. Correlation penalty: reduce weight if highly correlated with other selected assets
        std::vector<double> penalized_w = initial_w;
        for (int i = 0; i < n; ++i) {
            double penalty = 1.0;
            for (int j = 0; j < n; ++j) {
                if (i != j && std::abs(correlation_matrix[i][j]) > 0.6) {
                    // Penalty scales with correlation magnitude
                    penalty *= (1.0 - std::abs(correlation_matrix[i][j]) * 0.5);
                }
            }
            penalized_w[i] *= penalty;
        }

        // 3. Re-normalize to max_leverage and apply position limits
        double sum_abs_w = 0.0;
        for (double w : penalized_w) sum_abs_w += std::abs(w);
        
        if (sum_abs_w > 0.0) {
            double scale = max_leverage / sum_abs_w;
            for (int i = 0; i < n; ++i) {
                double w = penalized_w[i] * scale;
                // Clip to max position size
                if (w > max_position_size) w = max_position_size;
                if (w < -max_position_size) w = -max_position_size;
                
                // Drop negligible weights
                if (std::abs(w) < min_weight) w = 0.0;
                
                if (w != 0.0) {
                    weights[symbols[i]] = w;
                }
            }
        }

        return weights;
    }
};

void bind_optimization(py::module_& m) {
    py::class_<CppPortfolioOptimizer>(m, "CppPortfolioOptimizer")
        .def(py::init<double, double, double>(), 
             py::arg("max_position_size") = 0.2, 
             py::arg("max_leverage") = 1.0, 
             py::arg("min_weight") = 0.01)
        .def("optimize_weights", &CppPortfolioOptimizer::optimize_weights,
             py::arg("symbols"), py::arg("signals"), py::arg("correlation_matrix"));
}

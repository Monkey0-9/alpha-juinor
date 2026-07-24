#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class CppVectorizedBacktester {
private:
    double initial_capital;
    double transaction_cost;
    
public:
    CppVectorizedBacktester(double init_cap = 100000.0, double tc = 0.001)
        : initial_capital(init_cap), transaction_cost(tc) {}

    std::unordered_map<std::string, double> run_backtest(
        const std::vector<double>& prices,
        const std::vector<double>& signals) 
    {
        size_t n = prices.size();
        if (signals.size() != n) {
            throw std::invalid_argument("Prices and signals must have the same length.");
        }

        double capital = initial_capital;
        double position = 0.0;
        
        std::vector<double> equity_curve;
        equity_curve.reserve(n);
        
        double peak_capital = capital;
        double max_drawdown = 0.0;

        for (size_t i = 0; i < n; ++i) {
            if (i > 0) {
                // Mark to market
                double ret = (prices[i] - prices[i-1]) / prices[i-1];
                capital += position * capital * ret;
            }
            
            // Rebalance
            double target_position = signals[i];
            if (std::abs(target_position - position) > 0.01) {
                double trade_size = std::abs(target_position - position) * capital;
                capital -= trade_size * transaction_cost;
                position = target_position;
            }
            
            equity_curve.push_back(capital);
            
            if (capital > peak_capital) peak_capital = capital;
            double drawdown = (peak_capital - capital) / peak_capital;
            if (drawdown > max_drawdown) max_drawdown = drawdown;
        }

        double total_return = (capital - initial_capital) / initial_capital;
        
        std::unordered_map<std::string, double> results;
        results["final_capital"] = capital;
        results["total_return"] = total_return;
        results["max_drawdown"] = max_drawdown;
        
        return results;
    }
};

void bind_backtest(py::module_& m) {
    py::class_<CppVectorizedBacktester>(m, "CppVectorizedBacktester")
        .def(py::init<double, double>(), 
             py::arg("initial_capital") = 100000.0, 
             py::arg("transaction_cost") = 0.001)
        .def("run_backtest", &CppVectorizedBacktester::run_backtest,
             py::arg("prices"), py::arg("signals"));
}

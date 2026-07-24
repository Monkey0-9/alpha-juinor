#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct BarData {
    double open;
    double high;
    double low;
    double close;
    double volume;
};

class CppMarketDataFeed {
private:
    std::unordered_map<std::string, std::vector<BarData>> data_buffer;
    
public:
    CppMarketDataFeed() {}

    void ingest_bars(const std::string& symbol, 
                     const std::vector<double>& opens,
                     const std::vector<double>& highs,
                     const std::vector<double>& lows,
                     const std::vector<double>& closes,
                     const std::vector<double>& volumes) {
        
        size_t n = opens.size();
        if (highs.size() != n || lows.size() != n || closes.size() != n || volumes.size() != n) {
            throw std::invalid_argument("All arrays must be the same length.");
        }
        
        auto& buffer = data_buffer[symbol];
        for (size_t i = 0; i < n; ++i) {
            buffer.push_back({opens[i], highs[i], lows[i], closes[i], volumes[i]});
        }
    }

    std::vector<double> get_closes(const std::string& symbol) {
        if (data_buffer.find(symbol) == data_buffer.end()) return {};
        
        const auto& buffer = data_buffer[symbol];
        std::vector<double> closes;
        closes.reserve(buffer.size());
        for (const auto& bar : buffer) {
            closes.push_back(bar.close);
        }
        return closes;
    }
    
    std::vector<double> get_volumes(const std::string& symbol) {
        if (data_buffer.find(symbol) == data_buffer.end()) return {};
        
        const auto& buffer = data_buffer[symbol];
        std::vector<double> volumes;
        volumes.reserve(buffer.size());
        for (const auto& bar : buffer) {
            volumes.push_back(bar.volume);
        }
        return volumes;
    }
    
    void clear() {
        data_buffer.clear();
    }
};

void bind_feed(py::module_& m) {
    py::class_<CppMarketDataFeed>(m, "CppMarketDataFeed")
        .def(py::init<>())
        .def("ingest_bars", &CppMarketDataFeed::ingest_bars,
             py::arg("symbol"), py::arg("opens"), py::arg("highs"), 
             py::arg("lows"), py::arg("closes"), py::arg("volumes"))
        .def("get_closes", &CppMarketDataFeed::get_closes, py::arg("symbol"))
        .def("get_volumes", &CppMarketDataFeed::get_volumes, py::arg("symbol"))
        .def("clear", &CppMarketDataFeed::clear);
}

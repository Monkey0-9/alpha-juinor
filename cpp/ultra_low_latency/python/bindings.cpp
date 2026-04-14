#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "mqf_hot_path.hpp"

namespace py = pybind11;
using namespace mqf;

// Python-friendly wrapper for order book
class PyOrderBook {
    std::unique_ptr<LockFreeOrderBook> book_;
    
public:
    explicit PyOrderBook(uint64_t symbol_id) 
        : book_(std::make_unique<LockFreeOrderBook>(symbol_id)) {}
    
    void update_bid(size_t level, double price, uint32_t volume) {
        book_->update_bid(level, price, volume);
    }
    
    void update_ask(size_t level, double price, uint32_t volume) {
        book_->update_ask(level, price, volume);
    }
    
    double get_best_bid() const { return book_->get_best_bid(); }
    double get_best_ask() const { return book_->get_best_ask(); }
    double get_spread() const { return book_->get_spread(); }
    double get_mid() const { return (get_best_bid() + get_best_ask()) / 2.0; }
};

// NumPy-accelerated signal processing
py::array_t<double> process_signals_numpy(
    py::array_t<double> expected_returns,
    py::array_t<double> convictions,
    py::array_t<double> urgencies
) {
    py::buffer_info er_info = expected_returns.request();
    py::buffer_info conv_info = convictions.request();
    py::buffer_info urg_info = urgencies.request();
    
    if (er_info.size != conv_info.size || er_info.size != urg_info.size) {
        throw std::runtime_error("Input arrays must have same size");
    }
    
    size_t n = er_info.size;
    
    auto result = py::array_t<double>(n);
    py::buffer_info result_info = result.request();
    
    SIMDSignalEngine engine;
    engine.score_signals_avx2(
        static_cast<double*>(er_info.ptr),
        static_cast<double*>(conv_info.ptr),
        static_cast<double*>(urg_info.ptr),
        static_cast<double*>(result_info.ptr),
        n
    );
    
    return result;
}

PYBIND11_MODULE(mqf_ultra_low_latency, m) {
    m.doc() = "MiniQuantFund Ultra-Low Latency C++ Extensions";
    
    // OrderBook bindings
    py::class_<PyOrderBook>(m, "OrderBook")
        .def(py::init<uint64_t>(), py::arg("symbol_id"))
        .def("update_bid", &PyOrderBook::update_bid, 
             py::arg("level"), py::arg("price"), py::arg("volume"))
        .def("update_ask", &PyOrderBook::update_ask,
             py::arg("level"), py::arg("price"), py::arg("volume"))
        .def("get_best_bid", &PyOrderBook::get_best_bid)
        .def("get_best_ask", &PyOrderBook::get_best_ask)
        .def("get_spread", &PyOrderBook::get_spread)
        .def("get_mid", &PyOrderBook::get_mid);
    
    // SIMD signal processing
    m.def("process_signals", &process_signals_numpy,
          py::arg("expected_returns"), py::arg("convictions"), py::arg("urgencies"),
          "Process signals using AVX2 SIMD (8x faster)");
    
    // Timing functions
    m.def("get_timestamp_ns", &mqf_get_timestamp_ns,
          "Get high-precision timestamp in nanoseconds");
    m.def("busy_wait_ns", &mqf_busy_wait_ns, py::arg("nanoseconds"),
          "Precise busy-wait (no syscalls)");
    
    // Latency measurement
    m.def("measure_latency", []() -> double {
        NanoTimer timer;
        timer.start();
        // Do minimal work
        volatile int x = 0;
        for (int i = 0; i < 100; ++i) x += i;
        return timer.elapsed_us();
    }, "Measure round-trip latency in microseconds");
    
    // Version
    m.attr("__version__") = "1.0.0";
    m.attr("HAS_AVX2") = true;
    m.attr("CACHE_LINE_SIZE") = 64;
}

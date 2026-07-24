#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vector>
#include <cmath>
#include <deque>
#include <algorithm>
#include <numeric>
#include <limits>
#include <memory>
#include <functional>

namespace py = pybind11;

// ── Market Tick ───────────────────────────────────────────────────────
struct Tick {
    double price;
    double volume;
    uint64_t timestamp_ns;
    char side; // 'B', 'S', or 'T' for trade
};

// ── Order Book Level ──────────────────────────────────────────────────
struct Level {
    double price;
    double size;
    int order_count;
};

// ── Market Data Feed Handler ──────────────────────────────────────────
class MarketDataFeed {
public:
    std::string symbol;
    std::deque<double> prices;
    std::deque<double> volumes;
    std::deque<uint64_t> timestamps;
    size_t max_buffer;

    // Running calculations
    double kalman_state;
    double kalman_p;
    double vwap_cum_pv;
    double vwap_cum_v;
    double latest_vwap;
    double hurst_buffer[100];
    size_t hurst_idx;

    // Callbacks
    std::function<void(const Tick&)> on_tick_callback;
    std::function<void(double, double)> on_signal_callback;

    MarketDataFeed(const std::string& sym = "", size_t max_buf = 5000)
        : symbol(sym), max_buffer(max_buf), kalman_state(0.0), kalman_p(1.0),
          vwap_cum_pv(0.0), vwap_cum_v(0.0), latest_vwap(0.0), hurst_idx(0)
    {
        std::fill(hurst_buffer, hurst_buffer + 100, 0.0);
    }

    void on_tick(const Tick& tick) {
        prices.push_back(tick.price);
        volumes.push_back(tick.volume);
        timestamps.push_back(tick.timestamp_ns);

        if (prices.size() > max_buffer) {
            prices.pop_front();
            volumes.pop_front();
            timestamps.pop_front();
        }

        update_kalman(tick.price);
        update_vwap(tick.price, tick.volume);
        update_hurst(tick.price);
        update_hawkes(tick.price);

        if (on_tick_callback) on_tick_callback(tick);
        if (on_signal_callback && prices.size() >= 20) {
            double signal = compute_microstructure_signal();
            on_signal_callback(tick.price, signal);
        }
    }

    void update_kalman(double price) {
        if (kalman_state == 0.0) {
            kalman_state = price;
            return;
        }
        double q = 1e-5, r = 1e-4;
        kalman_p = kalman_p + q;
        double k = kalman_p / (kalman_p + r);
        kalman_state = kalman_state + k * (price - kalman_state);
        kalman_p = (1.0 - k) * kalman_p;
    }

    void update_vwap(double price, double volume) {
        vwap_cum_pv += price * volume;
        vwap_cum_v += volume;
        latest_vwap = (vwap_cum_v > 0) ? vwap_cum_pv / vwap_cum_v : price;
    }

    void update_hurst(double price) {
        if (prices.size() >= 2) {
            double ret = std::log(price / prices.back());
            hurst_buffer[hurst_idx % 100] = ret;
            hurst_idx++;
        }
    }

    void update_hawkes(double price) {
        // Track extreme events (|return| > 2 stdev)
        if (prices.size() < 20) return;

        double sum = 0.0, sq_sum = 0.0;
        int n = std::min(20, (int)prices.size());
        auto it = prices.end() - n;
        for (int i = 0; i < n - 1; ++i) {
            double r = (*(it + i + 1)) / (*(it + i)) - 1.0;
            sum += r; sq_sum += r * r;
        }
        double mean = sum / (n - 1);
        double stdev = std::sqrt(sq_sum / (n - 1) - mean * mean);

        double last_ret = price / *(prices.end() - 2) - 1.0;
        if (stdev > 0 && std::abs(last_ret) > 2.0 * stdev) {
            hawkes_excitation += 0.5;
        }
        hawkes_excitation *= 0.8; // decay
    }

    double compute_microstructure_signal() const {
        if (prices.size() < 20) return 0.0;

        // VWAP deviation
        double last_price = prices.back();
        double vwap_dev = (latest_vwap > 0) ? (last_price / latest_vwap - 1.0) : 0.0;

        // Kalman trend
        double kalman_trend = (kalman_state > 0) ? (last_price / kalman_state - 1.0) : 0.0;

        // Price velocity (5-tick)
        double vel = 0.0;
        if (prices.size() >= 6) {
            vel = (prices.back() - *(prices.end() - 6)) / *(prices.end() - 6);
        }

        // Order flow imbalance proxy (from volume)
        double volume_imbalance = 0.0;
        if (volumes.size() >= 5) {
            double recent_v = 0.0, older_v = 0.0;
            for (int i = 0; i < 5 && i < (int)volumes.size(); ++i) {
                if (i < 3) recent_v += *(volumes.end() - 1 - i);
                older_v += *(volumes.end() - 6 - i);
            }
            volume_imbalance = (older_v > 0) ? (recent_v / older_v - 1.0) : 0.0;
        }

        // Combined signal
        double signal = 0.35 * std::tanh(vwap_dev * 50.0) +
                        0.30 * std::tanh(kalman_trend * 100.0) +
                        0.20 * std::tanh(vel * 50.0) +
                        0.15 * std::tanh(volume_imbalance * 5.0);

        return std::max(-1.0, std::min(1.0, signal));
    }

    double get_kalman_state() const { return kalman_state; }
    double get_vwap() const { return latest_vwap; }
    double get_mid_price() const { return prices.empty() ? 0.0 : prices.back(); }
    double get_volatility() const {
        if (prices.size() < 10) return 0.0;
        double sum = 0.0, sq_sum = 0.0;
        int n = std::min(20, (int)prices.size());
        auto it = prices.end() - n;
        for (int i = 0; i < n - 1; ++i) {
            double r = (*(it + i + 1)) / (*(it + i)) - 1.0;
            sum += r; sq_sum += r * r;
        }
        return std::sqrt(sq_sum / (n - 1) - (sum / (n - 1)) * (sum / (n - 1)));
    }

private:
    double hawkes_excitation = 0.0;
};

// ── Signal Generator from Feed ────────────────────────────────────────
struct FeedSignal {
    double price;
    double signal;
    double vwap_deviation;
    double kalman_trend;
    double volatility;
    uint64_t timestamp;
};

FeedSignal generate_feed_signal(const MarketDataFeed& feed) {
    FeedSignal sig;
    sig.price = feed.get_mid_price();
    sig.signal = feed.compute_microstructure_signal();
    sig.vwap_deviation = (feed.get_vwap() > 0) ? (sig.price / feed.get_vwap() - 1.0) : 0.0;
    sig.kalman_trend = (feed.get_kalman_state() > 0) ? (sig.price / feed.get_kalman_state() - 1.0) : 0.0;
    sig.volatility = feed.get_volatility();
    sig.timestamp = feed.timestamps.empty() ? 0 : feed.timestamps.back();
    return sig;
}

// ── Pybind11 ──────────────────────────────────────────────────────────

PYBIND11_MODULE(nexus_feed, m) {
    m.doc() = "Nexus Real-Time Market Data Feed Handler";

    py::class_<Tick>(m, "Tick")
        .def(py::init<>())
        .def_readwrite("price", &Tick::price)
        .def_readwrite("volume", &Tick::volume)
        .def_readwrite("timestamp_ns", &Tick::timestamp_ns)
        .def_readwrite("side", &Tick::side);

    py::class_<FeedSignal>(m, "FeedSignal")
        .def_readonly("price", &FeedSignal::price)
        .def_readonly("signal", &FeedSignal::signal)
        .def_readonly("vwap_deviation", &FeedSignal::vwap_deviation)
        .def_readonly("kalman_trend", &FeedSignal::kalman_trend)
        .def_readonly("volatility", &FeedSignal::volatility)
        .def_readonly("timestamp", &FeedSignal::timestamp);

    py::class_<MarketDataFeed>(m, "MarketDataFeed")
        .def(py::init<const std::string&, size_t>(),
             py::arg("symbol") = "", py::arg("max_buffer") = 5000)
        .def("on_tick", &MarketDataFeed::on_tick)
        .def("compute_microstructure_signal", &MarketDataFeed::compute_microstructure_signal)
        .def("get_kalman_state", &MarketDataFeed::get_kalman_state)
        .def("get_vwap", &MarketDataFeed::get_vwap)
        .def("get_mid_price", &MarketDataFeed::get_mid_price)
        .def("get_volatility", &MarketDataFeed::get_volatility)
        .def_readwrite("symbol", &MarketDataFeed::symbol)
        .def_readwrite("on_tick_callback", &MarketDataFeed::on_tick_callback)
        .def_readwrite("on_signal_callback", &MarketDataFeed::on_signal_callback)
        .def_property_readonly("price_count", [](const MarketDataFeed& f) { return f.prices.size(); });

    m.def("generate_feed_signal", &generate_feed_signal, "Generate complete signal from feed state");
}
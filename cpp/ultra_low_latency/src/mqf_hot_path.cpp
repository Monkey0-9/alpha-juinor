#include "mqf_hot_path.hpp"
#include <thread>
#include <chrono>

namespace mqf {

// ============================================================================
// ORDER BOOK IMPLEMENTATION
// ============================================================================

extern "C" {

MQF_EXPORT void* mqf_create_orderbook(uint64_t symbol_id) {
    return new LockFreeOrderBook(symbol_id);
}

MQF_EXPORT void mqf_destroy_orderbook(void* handle) {
    delete static_cast<LockFreeOrderBook*>(handle);
}

MQF_EXPORT void mqf_update_bid(void* handle, size_t level, double price, uint32_t volume) {
    static_cast<LockFreeOrderBook*>(handle)->update_bid(level, price, volume);
}

MQF_EXPORT void mqf_update_ask(void* handle, size_t level, double price, uint32_t volume) {
    static_cast<LockFreeOrderBook*>(handle)->update_ask(level, price, volume);
}

MQF_EXPORT double mqf_get_spread(void* handle) {
    return static_cast<LockFreeOrderBook*>(handle)->get_spread();
}

MQF_EXPORT double mqf_get_mid(void* handle) {
    auto* book = static_cast<LockFreeOrderBook*>(handle);
    return (book->get_best_bid() + book->get_best_ask()) / 2.0;
}

// ============================================================================
// SIMD TICK PROCESSING
// ============================================================================

MQF_EXPORT void mqf_process_ticks_simd(
    const MarketTick* ticks,
    size_t count,
    double* signal_scores
) {
    SIMDSignalEngine engine;
    
    // Extract features from ticks
    alignas(32) std::array<double, 1024> returns;
    alignas(32) std::array<double, 1024> convictions;
    alignas(32) std::array<double, 1024> urgencies;
    
    size_t n = std::min(count, size_t(1024));
    
    for (size_t i = 0; i < n; ++i) {
        // Convert tick to signal features
        returns[i] = ticks[i].price * 0.001;  // Simplified return calc
        convictions[i] = ticks[i].volume > 1000 ? 0.9 : 0.5;
        urgencies[i] = (ticks[i].side == 2) ? 1.0 : 0.5;  // Trade urgency
    }
    
    engine.score_signals_avx2(returns.data(), convictions.data(), 
                              urgencies.data(), signal_scores, n);
}

// ============================================================================
// HIGH-PRECISION TIMING
// ============================================================================

MQF_EXPORT uint64_t mqf_get_timestamp_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

MQF_EXPORT void mqf_busy_wait_ns(uint64_t ns) {
    auto start = mqf_get_timestamp_ns();
    while ((mqf_get_timestamp_ns() - start) < ns) {
        _mm_pause();  // Hint to CPU we're spinning
    }
}

} // extern "C"

} // namespace mqf

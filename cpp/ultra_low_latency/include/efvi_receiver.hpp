#pragma once

#include <cstdint>
#include <atomic>
#include <array>
#include <immintrin.h>

#if defined(__x86_64__)
#define ALIGN_TO_CACHE_LINE alignas(64)
#else
#define ALIGN_TO_CACHE_LINE
#endif

namespace nexus {
namespace net {

/**
 * @brief Zero-copy UDP multicast receiver utilizing Solarflare EFVI (kernel bypass).
 * 
 * This class maps the NIC's ring buffer directly into userspace via hugepages (`memfd`),
 * avoiding context switches, memory copying, and interrupts. Polling is done in a tight
 * spin-loop utilizing PAUSE instructions to prevent pipeline stalls and thermal throttling.
 */
class ALIGN_TO_CACHE_LINE EfviMulticastReceiver {
public:
    EfviMulticastReceiver(const char* interface_name, uint16_t port);
    ~EfviMulticastReceiver();

    // Deleted copy constructors to prevent accidental expensive copies in the hot-path
    EfviMulticastReceiver(const EfviMulticastReceiver&) = delete;
    EfviMulticastReceiver& operator=(const EfviMulticastReceiver&) = delete;

    /**
     * @brief Polls the NIC's completion queue (CQ) for a new packet.
     * @return Pointer to the payload, or nullptr if no packet is ready.
     */
    inline const uint8_t* poll() noexcept {
        // Hot-path CQ polling
        // In a real implementation this directly accesses the mapped ef_eventq
        // For demonstration, simulating atomic acquire
        if (__builtin_expect(packet_ready_.load(std::memory_order_acquire), 0)) {
            packet_ready_.store(false, std::memory_order_relaxed);
            return rx_buffer_arena_.data();
        }
        
        // PAUSE instruction hints to the CPU that this is a spin-wait loop
        _mm_pause();
        return nullptr;
    }

private:
    ALIGN_TO_CACHE_LINE std::atomic<bool> packet_ready_{false};
    
    // Explicit padding to ensure rx_buffer_arena_ is on its own cache line
    // preventing false sharing with packet_ready_ during DMA writes from the NIC.
    uint8_t padding_[64 - sizeof(std::atomic<bool>)];

    ALIGN_TO_CACHE_LINE std::array<uint8_t, 2048> rx_buffer_arena_;
};

} // namespace net
} // namespace nexus

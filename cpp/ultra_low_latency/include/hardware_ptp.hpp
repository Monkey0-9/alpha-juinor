#pragma once

#include <cstdint>
#include <sys/socket.h>
#include <linux/net_tstamp.h>
#include <linux/errqueue.h>

namespace nexus {
namespace network {

/**
 * @brief Tower Research Style Hardware Timestamping (IEEE 1588 PTP)
 * 
 * Crucial for microwave network link arbitrage and precise event ordering.
 * Configures the NIC to extract the nanosecond timestamp exactly when the 
 * Start of Frame Delimiter (SFD) hits the physical MAC layer.
 */
class PtpMicrowaveClock {
public:
    static void enable_hardware_timestamps(int socket_fd) {
        int flags = SOF_TIMESTAMPING_RX_HARDWARE | 
                    SOF_TIMESTAMPING_RAW_HARDWARE | 
                    SOF_TIMESTAMPING_SYS_HARDWARE;
        
        setsockopt(socket_fd, SOL_SOCKET, SO_TIMESTAMPING, &flags, sizeof(flags));
    }

    /**
     * @brief Extracts the hardware timestamp from the control message ancillary data (cmsg).
     */
    static uint64_t extract_mac_layer_nanoseconds(struct msghdr* msg) noexcept {
        for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(msg, cmsg)) {
            if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_TIMESTAMPING) {
                // hw_timestamps[2] contains the raw hardware timestamp from the NIC
                struct timespec* hw_timestamps = reinterpret_cast<struct timespec*>(CMSG_DATA(cmsg));
                return static_cast<uint64_t>(hw_timestamps[2].tv_sec) * 1000000000ULL + 
                       static_cast<uint64_t>(hw_timestamps[2].tv_nsec);
            }
        }
        return 0; // Fallback or error handling in hot path
    }
};

} // namespace network
} // namespace nexus

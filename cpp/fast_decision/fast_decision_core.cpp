#include <cstddef>
#include <limits>

#ifdef _WIN32
#define MQF_EXPORT __declspec(dllexport)
#else
#define MQF_EXPORT __attribute__((visibility("default")))
#endif

namespace {
double holding_multiplier(int code) {
    switch (code) {
    case 0: return 2.0;  // microseconds
    case 1: return 1.5;  // seconds
    case 2: return 1.0;  // minutes
    case 3: return 0.7;  // hours
    case 4: return 0.5;  // days
    case 5: return 0.3;  // months
    default: return 0.5;
    }
}
}

extern "C" {
MQF_EXPORT double mqf_score_signal(
    double expected_return,
    double conviction,
    double urgency,
    int holding_code
) {
    const double risk_adjusted_return = expected_return * conviction;
    const double urgency_bonus = urgency * 0.5;
    return (risk_adjusted_return + urgency_bonus) * holding_multiplier(holding_code);
}

MQF_EXPORT int mqf_select_best_signal(
    const double* expected_returns,
    const double* convictions,
    const double* urgencies,
    const int* holding_codes,
    int n
) {
    if (!expected_returns || !convictions || !urgencies || !holding_codes || n <= 0) {
        return -1;
    }

    int best_index = 0;
    double best_score = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < n; ++i) {
        const double score = mqf_score_signal(
            expected_returns[i],
            convictions[i],
            urgencies[i],
            holding_codes[i]
        );
        if (score > best_score) {
            best_score = score;
            best_index = i;
        }
    }

    return best_index;
}
}

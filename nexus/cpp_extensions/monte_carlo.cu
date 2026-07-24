// ── GPU-Accelerated Monte Carlo VaR ──────────────────────────────────
// CUDA kernel for massively parallel Monte Carlo simulation.
// Compile with: nvcc -shared -o nexus_cuda.pyd monte_carlo.cu
// Falls back to OpenMP CPU when CUDA not available.

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ── CUDA Kernel: Monte Carlo VaR ─────────────────────────────────────
__global__ void mc_var_kernel(
    const float* historical_returns,
    int n_returns,
    float* path_results,
    int num_paths,
    int horizon,
    unsigned long long seed) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float path_sum = 0.0f;
    for (int d = 0; d < horizon; ++d) {
        int ri = curand(&state) % n_returns;
        path_sum += historical_returns[ri];
    }
    path_results[idx] = path_sum;
}

// ── CUDA Kernel: Survival Analysis ───────────────────────────────────
__global__ void survival_kernel(
    float initial_capital,
    float mu,
    float sigma,
    float ruin_level,
    int* survived,
    int days,
    int n_simulations,
    unsigned long long seed) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_simulations) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float capital = initial_capital;
    for (int d = 0; d < days; ++d) {
        float ret = mu + sigma * curand_normal(&state);
        capital *= (1.0f + ret);
        if (capital <= ruin_level) return;
    }
    atomicAdd(survived, 1);
}

#else
// CPU fallback when CUDA not available
// These are implemented in fast_math.cpp with OpenMP
#endif

// ── Host-side wrappers (compiled by both CUDA and CPU) ───────────────
#include <vector>
#include <algorithm>
#include <cmath>

#ifdef __CUDACC__
// CUDA host wrapper
float cuda_monte_carlo_var(
    const std::vector<float>& returns,
    int num_paths,
    int horizon,
    float confidence_level) {

    int n = returns.size();
    if (n == 0) return 0.0f;

    float *d_returns, *d_results;
    cudaMalloc(&d_returns, n * sizeof(float));
    cudaMalloc(&d_results, num_paths * sizeof(float));
    cudaMemcpy(d_returns, returns.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (num_paths + threads - 1) / threads;
    mc_var_kernel<<<blocks, threads>>>(d_returns, n, d_results, num_paths, horizon, 1337);

    std::vector<float> results(num_paths);
    cudaMemcpy(results.data(), d_results, num_paths * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_returns);
    cudaFree(d_results);

    std::sort(results.begin(), results.end());
    int idx = static_cast<int>((1.0f - confidence_level) * num_paths);
    idx = std::max(0, std::min(idx, num_paths - 1));
    return results[idx];
}

float cuda_survival_analysis(
    float initial_capital,
    float mu,
    float sigma,
    int days,
    int n_simulations,
    float ruin_threshold) {

    float *d_ruin_level;
    int *d_survived, h_survived = 0;
    float ruin_level = initial_capital * (1.0f - ruin_threshold);

    cudaMalloc(&d_survived, sizeof(int));
    cudaMemcpy(d_survived, &h_survived, sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n_simulations + threads - 1) / threads;
    survival_kernel<<<blocks, threads>>>(
        initial_capital, mu, sigma, ruin_level, d_survived, days, n_simulations, 1337);

    cudaMemcpy(&h_survived, d_survived, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_survived);

    return static_cast<float>(h_survived) / n_simulations;
}
#else
// CPU stub - actual implementation is in fast_math.cpp with OpenMP
float cuda_monte_carlo_var(
    const std::vector<float>& returns,
    int num_paths,
    int horizon,
    float confidence_level) { return 0.0f; }

float cuda_survival_analysis(
    float initial_capital, float mu, float sigma,
    int days, int n_simulations, float ruin_threshold) { return 0.0f; }
#endif

// ── Pybind11 bindings ─────────────────────────────────────────────────
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(nexus_cuda, m) {
    m.doc() = "Nexus GPU-Accelerated Monte Carlo Kernels";

    m.def("monte_carlo_var", &cuda_monte_carlo_var,
        "GPU-accelerated Monte Carlo VaR (CUDA)",
        py::arg("returns"), py::arg("num_paths") = 100000,
        py::arg("horizon") = 20, py::arg("confidence_level") = 0.95);

    m.def("survival_analysis", &cuda_survival_analysis,
        "GPU-accelerated survival analysis (CUDA)",
        py::arg("initial_capital"), py::arg("mu"), py::arg("sigma"),
        py::arg("days") = 252, py::arg("n_simulations") = 100000,
        py::arg("ruin_threshold") = 0.5);

    m.attr("CUDA_AVAILABLE") =
#ifdef __CUDACC__
        true;
#else
        false;
#endif
}
/**
 *  @file src/smallfield/cuda_error.hpp
 *  @brief Small sink that routes CUDA / cuSOLVER error reports through a `logger_t`.
 *
 *  Bumps an owner-supplied `std::atomic` counter. Pair up a `logger_t` and an error counter, both typically fields
 *  of `context_t`, into one `cuda_error_sink_t` and pass it by value to every function that issues CUDA API calls.
 *  Sink is a plain struct; both fields are pointers to state owned elsewhere.
 */

#pragma once

#include <cstdint> // `std::uint64_t`

#include <atomic> // `std::atomic`

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "smallfield/log.hpp"

namespace smallfield {

/**
 *  @brief CUDA error-reporting sink: logger + counter pointers, owned elsewhere.
 *
 *  Default-constructed sinks are benign: `check` silently counts successes and drops errors on the floor. The
 *  intended usage is to construct via `{.log = &context.logger, .count = &context.cuda_error_count}` and pass
 *  by value down the call chain.
 */
struct cuda_error_sink_t {
    logger_t const *log = nullptr;
    std::atomic<std::uint64_t> *count = nullptr;

    /** Log and count on non-cudaSuccess. Returns true iff status == cudaSuccess. */
    bool check(cudaError_t status, char const *label) const noexcept {
        if (status == cudaSuccess) return true;
        if (log != nullptr) log->error("CUDA error in {}: {}", label, cudaGetErrorString(status));
        if (count != nullptr) count->fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    /** Log and count on non-CUSOLVER_STATUS_SUCCESS. Returns true on success. */
    bool check(cusolverStatus_t status, char const *label) const noexcept {
        if (status == CUSOLVER_STATUS_SUCCESS) return true;
        if (log != nullptr) log->error("cuSOLVER error in {}: {}", label, static_cast<int>(status));
        if (count != nullptr) count->fetch_add(1, std::memory_order_relaxed);
        return false;
    }
};

} // namespace smallfield

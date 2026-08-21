/**
 *  @file   src/smallfield/types/storage.hpp
 *  @brief  Owning storage, allocated and released in stream order.
 *
 *  `types/views.hpp` names shapes nobody owns; this header owns the memory and hands out those views, kept
 *  apart because owning storage drags in the CUDA runtime a shape-only header must not pay for.
 *
 *  Every allocation is stream-ordered and the allocator carries the stream. `cudaMalloc` and `cudaFree`
 *  synchronize the whole device, so under the per-device OpenMP fan-out one allocation serializes against
 *  every other device's in-flight work, while `cudaMallocAsync` draws from a per-device pool and orders
 *  against one stream that also @b carries the device, so a free reaches the context that made the
 *  allocation with no device id stored and switched to by hand. That removes the cost a hand-carved slab
 *  amortizes, leaving sub-allocation worth keeping only where one allocation makes an upload a single
 *  `cudaMemcpyAsync` — the wave pool, which carves itself, and nothing else here. `try_reserve` says only
 *  whether the buffer grew, a stream-ordered allocation's reason for failing belonging to the stream and
 *  read at the caller's existing sync point, while `upload_into` and `download_from` take a sink because a
 *  `cudaMemcpyAsync` argument error is a programming mistake worth naming immediately.
 */

#pragma once

#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <concepts> // `std::convertible_to`, `std::same_as`
#include <utility>  // `std::exchange`

#include <cuda_runtime.h>

#include "smallfield/cuda_error.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::types {

#pragma region Policies

/** Where an allocation lives, and therefore who may dereference it. A compile-time property read by
 *  `requires`-clauses, never by a branch on the allocation path: host code dereferencing a device pointer
 *  is the failure this makes unspellable, where a runtime enum inside the buffer would only let it be
 *  detected, one wave too late. */
enum class memory_residence_t : std::uint8_t {
    /** Reachable by the CPU. Pinned host memory is still host memory. */
    host_k,
    /** Reachable only by a kernel, and only through the stream that allocated it. */
    device_k,
};

/** What `try_reserve` did, and what the caller must do about interior pointers. A `bool` would fold
 *  "already large enough", the steady-state answer making no driver call, into the same `true` as "grew",
 *  an allocation on the path pre-reservation exists to keep clear; naming them apart makes that invariant
 *  assertable rather than a comment nothing re-checks. */
enum class reserve_outcome_t : std::uint8_t {
    /** Already large enough. Nothing allocated, every interior pointer still valid. */
    unchanged_k,
    /** Reallocated. Contents discarded and every interior pointer stale. */
    grown_k,
    /** Nothing allocated and the previous allocation is intact. The stream carries the reason. */
    failed_k,
};

/** Whether a table an uploader was asked to stage is readable by a kernel. A length cannot carry this: a
 *  legitimately empty table and a refused reservation or copy both yield an empty span, one meaning "no
 *  entry" to every lookup and the other that the caller holds no parameters at all. They call for opposite
 *  responses, so the uploader answers with a state and the caller drops the device on `refused_k`. */
enum class table_residency_t : std::uint8_t {
    /** Every table landed. An empty span among them is an empty table, not a lost one. */
    resident_k,
    /** A reservation or a copy was refused. Nothing may read the view. The stream carries the reason. */
    refused_k,
};

/** Acquires and releases raw bytes, returning the pointer and leaving the reason to the stream. `allocate`
 *  is not `const`, because a stream-ordered allocator may record the pool it drew from. */
template <typename allocator_type_>
concept memory_allocator = requires(allocator_type_ allocator, void *pointer, std::size_t bytes) {
    { allocator_type_::residence_k } -> std::convertible_to<memory_residence_t>;
    { allocator.allocate(bytes) } noexcept -> std::same_as<void *>;
    { allocator.deallocate(pointer) } noexcept;
};

/**
 *  @brief Page-locked host memory, so a `cudaMemcpyAsync` out of it is genuinely asynchronous.
 *
 *  A copy from pageable memory degrades to a synchronous staging copy inside the driver, which is why the
 *  wave pool is one pinned allocation rather than a `std::vector`. There is no stream here and that is no
 *  omission: `cudaHostAlloc` has no stream-ordered form — pinning pages is a host-side kernel operation — so
 *  carrying one would suggest an ordering the call lacks. Pinned buffers are reserved once at construction,
 *  off the wave path, where blocking is free.
 */
struct pinned_allocator_t {
    static constexpr memory_residence_t residence_k = memory_residence_t::host_k;

    void *allocate(std::size_t bytes) noexcept {
        void *raw = nullptr;
        return cudaHostAlloc(&raw, bytes, cudaHostAllocDefault) == cudaSuccess ? raw : nullptr;
    }
    void deallocate(void *pointer) noexcept { [[maybe_unused]] cudaError_t const status = cudaFreeHost(pointer); }
};

/** Device memory drawn from the per-device pool, allocated and freed against `stream`, whose error state
 *  reports their failure. That stream must outlive the buffer: `cudaFreeAsync` against a destroyed stream is
 *  undefined, so a buffer built on a per-wave stream is released before the stream is — the lifetime rule
 *  every other stream-ordered resource in the pipeline already follows. */
struct device_allocator_t {
    static constexpr memory_residence_t residence_k = memory_residence_t::device_k;

    cudaStream_t stream = nullptr;

    void *allocate(std::size_t bytes) noexcept {
        void *raw = nullptr;
        [[maybe_unused]] cudaError_t const status = cudaMallocAsync(&raw, bytes, stream);
        return raw;
    }
    void deallocate(void *pointer) noexcept {
        [[maybe_unused]] cudaError_t const status = cudaFreeAsync(pointer, stream);
    }
};

#pragma endregion Policies

#pragma region Stream

/** Owning CUDA stream, so the ordering `device_allocator_t` depends on is a lifetime rather than a sequence
 *  of statements. A raw `cudaStream_t` destroyed by hand leaves every buffer built on it at the mercy of
 *  statement order in some destructor's body, and a member freed after that body runs is a `cudaFreeAsync`
 *  against a destroyed stream. Owning it moves the question to member declaration order, which the compiler
 *  settles — declare the stream before the buffers ordered against it and it is destroyed after them. */
class device_stream_t {
    cudaStream_t handle_ = nullptr;

  public:
    device_stream_t() = default;
    device_stream_t(device_stream_t const &) = delete;
    device_stream_t &operator=(device_stream_t const &) = delete;
    device_stream_t(device_stream_t &&other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    device_stream_t &operator=(device_stream_t &&other) noexcept {
        if (this != &other) {
            release_();
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }
    ~device_stream_t() noexcept { release_(); }

    /** Create the stream on the current device, replacing whatever this held, and answer whether it exists
     *  afterward; a refusal leaves this empty and reaches @p sink's logger and error counter. */
    [[nodiscard]] bool create(cuda_error_sink_t const &sink = {}) noexcept {
        release_();
        return sink.check(cudaStreamCreate(&handle_), "cudaStreamCreate");
    }

    /** The borrowed handle every stream-ordered call takes, null until `create` succeeds. */
    [[nodiscard]] cudaStream_t handle() const noexcept { return handle_; }

  private:
    void release_() noexcept {
        if (handle_ == nullptr) return;
        [[maybe_unused]] cudaError_t const status = cudaStreamDestroy(std::exchange(handle_, nullptr));
    }
};

#pragma endregion Stream

#pragma region Buffer

/** Grow-only typed storage over one allocator. Contents do not survive a grow, and a grow that fails
 *  changes nothing: the replacement is acquired before the old allocation is released, so out of memory
 *  costs a wave rather than the working buffer the next wave would have reused. Not copyable, because a copy
 *  would have to choose between sharing an allocation two destructors will free and silently issuing a
 *  second one the size of the first. The allocator moves with the pointer, holding the stream that decides
 *  which device and which ordering the free runs against; leaving it behind leaks across devices. */
template <typename value_type_, memory_allocator allocator_type_>
class buffer {
    value_type_ *pointer_ = nullptr;
    std::size_t capacity_ = 0;
    [[no_unique_address]] allocator_type_ allocator_ {};

  public:
    using value_type = value_type_;
    using allocator_type = allocator_type_;
    static constexpr memory_residence_t residence_k = allocator_type_::residence_k;

    buffer() noexcept = default;
    /** Bind the allocator, and with it the stream every allocation and free will be ordered against. */
    explicit buffer(allocator_type_ allocator) noexcept : allocator_(allocator) {}
    ~buffer() noexcept { release_(); }

    buffer(buffer const &) = delete;
    buffer &operator=(buffer const &) = delete;

    buffer(buffer &&other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)), capacity_(std::exchange(other.capacity_, 0)),
          allocator_(std::exchange(other.allocator_, allocator_type_ {})) {}

    buffer &operator=(buffer &&other) noexcept {
        if (this == &other) return *this;
        release_();
        pointer_ = std::exchange(other.pointer_, nullptr);
        capacity_ = std::exchange(other.capacity_, 0);
        allocator_ = std::exchange(other.allocator_, allocator_type_ {});
        return *this;
    }

    /** Elements the buffer spans, which for a `std::byte` specialization are bytes. */
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] value_type_ *data() noexcept { return pointer_; }
    [[nodiscard]] value_type_ const *data() const noexcept { return pointer_; }

    /** The whole allocation as one view, so the extent never travels beside the pointer. */
    [[nodiscard]] custd::span<value_type_> view() noexcept { return {pointer_, capacity_}; }
    [[nodiscard]] custd::span<value_type_ const> view() const noexcept { return {pointer_, capacity_}; }

    /** Grow to span at least @p count elements — elements, not bytes — allocating before releasing.
     *  Answers what happened, and with it whether interior pointers survived; on `failed_k` the stream
     *  carries the reason, read at whatever sync point the caller already has. */
    [[nodiscard]] reserve_outcome_t try_reserve(std::size_t count) noexcept {
        if (count <= capacity_) return reserve_outcome_t::unchanged_k;

        void *raw = allocator_.allocate(count * sizeof(value_type_));
        if (raw == nullptr) return reserve_outcome_t::failed_k;

        release_();
        pointer_ = static_cast<value_type_ *>(raw);
        capacity_ = count;
        return reserve_outcome_t::grown_k;
    }

  private:
    void release_() noexcept {
        if (pointer_ == nullptr) return;
        allocator_.deallocate(pointer_);
        pointer_ = nullptr;
        capacity_ = 0;
    }
};

#pragma endregion Buffer

#pragma region Specializations

/** Page-locked staging for one wave, uploaded to the device in a single `cudaMemcpyAsync`. */
using pinned_bytes_t = buffer<std::byte, pinned_allocator_t>;

/** The device side of that same wave, carved into regions by the pool layout. */
using device_bytes_t = buffer<std::byte, device_allocator_t>;

/** A homogeneous device array, where the element type is known and no region carving happens. */
using device_indices_t = buffer<std::uint32_t, device_allocator_t>;

#pragma endregion Specializations

#pragma region Transfers

/**
 *  @brief Copy host elements onto the device along @p stream.
 *
 *  Both extents come from the spans, so a caller cannot restate one and get it wrong, and a destination
 *  smaller than the source is refused rather than truncated. A pageable source silently degrades this to a
 *  blocking staged copy inside the driver, which is why the wave pool stages through `pinned_bytes_t`; the
 *  call belongs on an explicit stream either way, the legacy default one serializing against every other.
 *  @param[out] destination Device elements to overwrite.
 *  @param[in] source Host elements to read.
 *  @param[in] stream Orders this copy against the rest of the wave's work.
 *  @param[in] sink Routes a rejected copy to the logger and the error counter.
 *  @return Whether the copy was enqueued. Completion is the stream's business, not this call's.
 */
template <typename element_>
[[nodiscard]] bool upload_into(custd::span<element_> destination, custd::span<element_ const> source,
                               cudaStream_t stream, cuda_error_sink_t const &sink = {}) noexcept {
    if (source.empty()) return true;
    if (destination.size() < source.size()) return sink.check(cudaErrorInvalidValue, "upload_into");
    return sink.check(
        cudaMemcpyAsync(destination.data(), source.data(), source.size_bytes(), cudaMemcpyHostToDevice, stream),
        "upload_into");
}

/**
 *  @brief Copy device elements back to the host along @p stream.
 *  @param[out] destination Host elements to overwrite, pinned if the copy is to overlap anything.
 *  @param[in] source Device elements to read.
 *  @param[in] stream Orders this copy after the kernels that produced @p source.
 *  @param[in] sink Routes a rejected copy to the logger and the error counter.
 *  @return Whether the copy was enqueued. The data is not there until the stream says so.
 */
template <typename element_>
[[nodiscard]] bool download_from(custd::span<element_> destination, custd::span<element_ const> source,
                                 cudaStream_t stream, cuda_error_sink_t const &sink = {}) noexcept {
    if (source.empty()) return true;
    if (destination.size() < source.size()) return sink.check(cudaErrorInvalidValue, "download_from");
    return sink.check(
        cudaMemcpyAsync(destination.data(), source.data(), source.size_bytes(), cudaMemcpyDeviceToHost, stream),
        "download_from");
}

#pragma endregion Transfers

} // namespace smallfield::types

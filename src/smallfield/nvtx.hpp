/**
 *  @file src/smallfield/nvtx.hpp
 *  @brief Scoped NVTX ranges for attributing host-side pipeline time in Nsight Systems.
 *
 *  The batch driver overlaps four roles — reader, gpu-runner, fallback, writer — on one thread each, so a flat
 *  `nsys` timeline shows CUDA activity but cannot say which role a host thread was in, nor which roles were
 *  genuinely concurrent. Wrapping each role's per-batch body in a named range makes the overlap, or lack of it,
 *  directly readable, which is what distinguishes "the GPU is idle because the pipeline stalled" from "the GPU
 *  is idle because the host is still doing RDKit work".
 *
 *  Header-only and dependency-free: NVTX3 ships inside the CUDA toolkit and needs no link-time library. Compiles
 *  to nothing when `SMALLFIELD_ENABLE_NVTX` is undefined, so annotated builds and shipping builds stay
 *  byte-identical by default.
 */
#pragma once

#if defined(SMALLFIELD_ENABLE_NVTX)

#include <cstdint> // `std::uint32_t`

#include <nvtx3/nvToolsExt.h>

namespace smallfield {

/** RAII NVTX range: pushes on construction, pops at end of scope. */
struct nvtx_range_t {
    explicit nvtx_range_t(char const *name, std::uint32_t color = 0x4080C0u) {
        nvtxEventAttributes_t attributes = {};
        attributes.version = NVTX_VERSION;
        attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attributes.colorType = NVTX_COLOR_ARGB;
        attributes.color = 0xFF000000u | color;
        attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attributes.message.ascii = name;
        nvtxRangePushEx(&attributes);
    }
    ~nvtx_range_t() { nvtxRangePop(); }
    nvtx_range_t(nvtx_range_t const &) = delete;
    nvtx_range_t &operator=(nvtx_range_t const &) = delete;
};

} // namespace smallfield

/**
 *  @brief `##` suppresses expansion of its operands, so `nvtx_scope_##__LINE__` would paste the
 *  literal identifier `nvtx_scope___LINE__` and collide whenever two ranges share a scope.
 *
 *  Two-level indirection expands `__LINE__` first.
 */
#define SF_NVTX_PASTE_(prefix, suffix) prefix##suffix
#define SF_NVTX_NAME_(line) SF_NVTX_PASTE_(nvtx_scope_, line)
#define SF_NVTX_RANGE_COLORED(name, color) \
    ::smallfield::nvtx_range_t SF_NVTX_NAME_(__LINE__) { name, color }

#else

#define SF_NVTX_RANGE_COLORED(name, color) ((void)0)

#endif

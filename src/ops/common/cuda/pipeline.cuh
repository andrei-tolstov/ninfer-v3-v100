#pragma once

#include "ops/common/cuda/memory.cuh"

#include <cuda.h>
#include <cuda_pipeline.h>

#include <cstdint>

namespace ninfer::ops::common {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes));
    }
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes), "r"(src_bytes));
    }
}

template <Cache Policy>
__device__ __forceinline__ void cp_async_cache_hint(void* destination, const void* source,
                                                    int valid_bytes, std::uint64_t policy) {
    if constexpr (Policy == Cache::cg) {
        asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                     :
                     : "r"(smem_addr(destination)), "l"(source), "r"(valid_bytes), "l"(policy));
    } else {
        asm volatile("cp.async.ca.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                     :
                     : "r"(smem_addr(destination)), "l"(source), "r"(valid_bytes), "l"(policy));
    }
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
}

__device__ __forceinline__ void pipe_commit() { __pipeline_commit(); }

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
    __pipeline_wait_prior(Groups);
}

#else // __CUDA_ARCH__ < 800 (Volta sm_70 fallback)

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    if constexpr (Bytes == 4) {
        *static_cast<unsigned*>(smem_dst) = *static_cast<const unsigned*>(gmem_src);
    } else if constexpr (Bytes == 8) {
        *static_cast<uint2*>(smem_dst) = *static_cast<const uint2*>(gmem_src);
    } else {
        *static_cast<uint4*>(smem_dst) = *static_cast<const uint4*>(gmem_src);
    }
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    auto* dst       = static_cast<unsigned char*>(smem_dst);
    const auto* src = static_cast<const unsigned char*>(gmem_src);
#pragma unroll
    for (int i = 0; i < Bytes; ++i) {
        dst[i] = (i < src_bytes) ? src[i] : static_cast<unsigned char>(0);
    }
}

template <Cache Policy>
__device__ __forceinline__ void cp_async_cache_hint(void* destination, const void* source,
                                                    int valid_bytes, std::uint64_t) {
    auto* dst       = static_cast<unsigned char*>(destination);
    const auto* src = static_cast<const unsigned char*>(source);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        dst[i] = (i < valid_bytes) ? src[i] : static_cast<unsigned char>(0);
    }
}

__device__ __forceinline__ void cp_commit() {}

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
    if constexpr (Bytes == 4) {
        *static_cast<unsigned*>(smem_dst) = *static_cast<const unsigned*>(gmem_src);
    } else if constexpr (Bytes == 8) {
        *static_cast<uint2*>(smem_dst) = *static_cast<const uint2*>(gmem_src);
    } else {
        *static_cast<uint4*>(smem_dst) = *static_cast<const uint4*>(gmem_src);
    }
}

__device__ __forceinline__ void pipe_commit() {}

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
}

#endif // __CUDA_ARCH__ < 800

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

__device__ __forceinline__ void cta_mbarrier_init(std::uint64_t* barrier, std::uint32_t arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(arrivals)
                 : "memory");
}

__device__ __forceinline__ void cta_mbarrier_wait(std::uint64_t* barrier, std::uint32_t phase) {
    constexpr std::uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n"
                 ".reg .pred done;\n"
                 "wait_loop_%=: \n"
                 "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
                 "@done bra wait_done_%=;\n"
                 "bra wait_loop_%=;\n"
                 "wait_done_%=: \n"
                 "}\n"
                 :
                 : "r"(smem_addr(barrier)), "r"(phase), "r"(kSuspendTicks)
                 : "memory");
}

__device__ __forceinline__ void cta_mbarrier_arrive(std::uint64_t* barrier) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(smem_addr(barrier)) : "memory");
}

__device__ __forceinline__ void cta_mbarrier_arrive_expect_tx(std::uint64_t* barrier,
                                                              std::uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void cta_mbarrier_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
}

#else // __CUDA_ARCH__ < 800

__device__ __forceinline__ void cta_mbarrier_init(std::uint64_t*, std::uint32_t) { __trap(); }
__device__ __forceinline__ void cta_mbarrier_wait(std::uint64_t*, std::uint32_t) { __trap(); }
__device__ __forceinline__ void cta_mbarrier_arrive(std::uint64_t*) { __trap(); }
__device__ __forceinline__ void cta_mbarrier_arrive_expect_tx(std::uint64_t*, std::uint32_t) { __trap(); }
__device__ __forceinline__ void cta_mbarrier_fence_init() {}

#endif // __CUDA_ARCH__ < 800

// The caller owns tensor-map geometry, shared layout, and mbarrier transaction accounting.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900

__device__ __forceinline__ void tma_load_2d(void* destination, const CUtensorMap* map, int c0,
                                            int c1, std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(c0), "r"(c1), "r"(smem_addr(barrier))
                 : "memory");
}

__device__ __forceinline__ void tma_load_3d(void* destination, const CUtensorMap* map, int c0,
                                            int c1, int c2, std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4}], [%5];"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(c0), "r"(c1), "r"(c2),
                   "r"(smem_addr(barrier))
                 : "memory");
}

__device__ __forceinline__ void tma_load_4d(void* destination, const CUtensorMap* map, int c0,
                                            int c1, int c2, int c3, std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4, %5}], [%6];"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
                   "r"(smem_addr(barrier))
                 : "memory");
}

#else // __CUDA_ARCH__ < 900

__device__ __forceinline__ void tma_load_2d(void*, const CUtensorMap*, int, int, std::uint64_t*) { __trap(); }
__device__ __forceinline__ void tma_load_3d(void*, const CUtensorMap*, int, int, int, std::uint64_t*) { __trap(); }
__device__ __forceinline__ void tma_load_4d(void*, const CUtensorMap*, int, int, int, int, std::uint64_t*) { __trap(); }

#endif // __CUDA_ARCH__ < 900

} // namespace ninfer::ops::common

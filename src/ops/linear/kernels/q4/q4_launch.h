#pragma once

#include "core/arena.h"
#include "core/weight.h"
#include "core/tensor.h"
#include "core/platform.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

using Q4Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

#if NINFER_TARGET_VOLTA
[[nodiscard]] bool q4_uses_volta_qpn(std::int32_t n, std::int32_t k, std::int32_t t) noexcept;
void launch_q4_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

[[nodiscard]] bool q4_volta_mma_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept;
void launch_q4_volta_mma(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                         cudaStream_t stream, int splits_override = 0);
[[nodiscard]] std::size_t q4_volta_mma_workspace_bytes(std::int32_t n, std::int32_t k,
                                                       std::int32_t t) noexcept;
#endif // NINFER_TARGET_VOLTA

} // namespace ninfer::ops::detail

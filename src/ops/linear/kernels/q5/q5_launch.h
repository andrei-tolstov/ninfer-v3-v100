#pragma once

#include "core/arena.h"
#include "core/weight.h"
#include "core/tensor.h"
#include "core/platform.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

using Q5Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

#if NINFER_TARGET_VOLTA
[[nodiscard]] bool q5_volta_mma_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept;
[[nodiscard]] std::size_t q5_volta_mma_workspace_bytes(std::int32_t n, std::int32_t k,
                                                       std::int32_t t) noexcept;
void launch_q5_volta_mma(const Tensor& x, const Weight& w, Tensor& out, bool add_residual,
                         std::int32_t weight_row_offset, WorkspaceArena& ws, cudaStream_t stream,
                         int splits_override = 0);
#endif // NINFER_TARGET_VOLTA

} // namespace ninfer::ops::detail

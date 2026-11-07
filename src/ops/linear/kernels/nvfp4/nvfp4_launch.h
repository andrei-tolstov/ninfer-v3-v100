#pragma once

#include "core/arena.h"
#include "core/weight.h"
#include "core/tensor.h"
#include "core/platform.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

using Nvfp4Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_nvfp4_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_nvfp4_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

#if NINFER_TARGET_VOLTA
inline constexpr std::int32_t kNvfp4VoltaQpnRowsPerTile = 8;
inline constexpr std::int32_t kNvfp4VoltaQpnMaxTokens   = 32;
void launch_nvfp4_volta_qpn(const Tensor&, const Weight&, Tensor&, cudaStream_t);
[[nodiscard]] bool nvfp4_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                             std::int32_t t) noexcept;

void launch_nvfp4_volta_mma(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                            cudaStream_t stream, int splits_override = 0);
[[nodiscard]] bool nvfp4_volta_mma_supported(std::int32_t n, std::int32_t k,
                                             std::int32_t t) noexcept;
[[nodiscard]] std::size_t nvfp4_volta_mma_workspace_bytes(std::int32_t n, std::int32_t k,
                                                          std::int32_t t) noexcept;
[[nodiscard]] int nvfp4_volta_mma_splits(std::int32_t n, std::int32_t k, std::int32_t t) noexcept;
#endif // NINFER_TARGET_VOLTA

} // namespace ninfer::ops::detail

#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "core/platform.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

using Q8Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
using W8Launch = Q8Launch;

#if NINFER_TARGET_VOLTA
void launch_w8_volta_mma(const Tensor&, const Weight&, Tensor&, cudaStream_t);
[[nodiscard]] bool w8_volta_mma_supported(std::int32_t n, std::int32_t k,
                                          std::int32_t t) noexcept;

void launch_w8_volta_qpn(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_volta_qpn_dynamic_conv_add(const Tensor& x, const Weight& weight,
                                          const Tensor& base, const Tensor& finish_delta,
                                          Tensor& residual, std::int32_t width,
                                          cudaStream_t stream);
[[nodiscard]] bool w8_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                          std::int32_t t) noexcept;
#endif // NINFER_TARGET_VOLTA

} // namespace ninfer::ops::detail

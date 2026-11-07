#include "ops/linear/projection/fp8/fp8_dispatch.h"
#include "ops/common/host/operands.h"
#include "ops/linear/projection/fp8/fp8_shapes.h"
#include "core/platform.h"
#include <array>
#include <stdexcept>

#if NINFER_TARGET_VOLTA
#include "ops/linear/kernels/fp8/fp8_cutlass_sm70.h"
#include "ops/linear/kernels/fp8/fp8_volta_qpn_gemm.cuh"
#endif

namespace ninfer::ops::detail {

#if NINFER_TARGET_VOLTA

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k, LinearPolicy,
                                                std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("fp8 linear workspace: invalid token interval");
    const std::size_t cutlass_bytes = fp8_cutlass_sm70_workspace_bytes(n, k, max_tokens);
    const std::size_t qpn_bytes     = static_cast<std::size_t>(k) * std::min(max_tokens, 32) * sizeof(std::uint16_t);
    return std::max(cutlass_bytes, qpn_bytes);
}

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy,
                  MemoryArena* workspace, cudaStream_t stream) {
    common::validate_fp8_weight(weight, "fp8 linear");
    const std::int32_t t = x.ne[1];
    if (t <= 0) throw std::invalid_argument("fp8 linear: T must be positive");

    if (t > 32 || !fp8_volta_qpn_supported(weight.n, weight.k, t)) {
        if (workspace != nullptr) {
            fp8_cutlass_sm70_launch(x, weight, out, *workspace, stream);
        } else {
            MemorySpan empty_storage{nullptr, 0, Device::cpu()};
            MemoryArena dummy(empty_storage);
            fp8_cutlass_sm70_launch(x, weight, out, dummy, stream);
        }
        return;
    }

    if (workspace != nullptr) {
        auto scope      = workspace->scope();
        auto activation = workspace->alloc_bytes(
            static_cast<std::size_t>(weight.k) * t * sizeof(std::uint16_t), 256);
        fp8_stage_bf16_activation_sm70(x, activation.data, stream);
        launch_fp8_volta_qpn_fp16(x, weight, activation.data, out, stream);
    } else {
        launch_fp8_volta_qpn(x, weight, out, stream);
    }
}

#else

namespace {
const std::array kShapes{&kFp8N14336K5120, &kFp8N16384K5120, &kFp8N34816K5120,
                         &kFp8N5120K6144,  &kFp8N5120K17408, &kFp8N248320K5120};

const Fp8LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("fp8 linear: unsupported policy");
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return *shape;
    throw std::invalid_argument("fp8 linear: unsupported shape");
}
} // namespace

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                                std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("fp8 linear workspace: invalid token interval");
    const auto& shape = resolve_shape(n, k, policy);
    return allows_a8(policy) && shape.uses_a8(min_tokens, max_tokens)
               ? fp8_a8_workspace_capacity_bytes(
                     max_tokens, k,
                     shape.partial_capacity_bytes ? shape.partial_capacity_bytes(max_tokens) : 0)
               : 0;
}

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  MemoryArena* workspace, cudaStream_t stream) {
    common::validate_fp8_weight(weight, "fp8 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("fp8 linear: T must be positive");
    const auto& shape = resolve_shape(weight.n, weight.k, policy);
    if (!allows_a8(policy) || !shape.uses_a8(x.ne[1], x.ne[1]))
        return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("fp8 A8 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_fp8_a8_workspace(
        *workspace, x.ne[1], weight.k,
        shape.partial_capacity_bytes ? shape.partial_capacity_bytes(x.ne[1]) : 0);
    shape.a8(x, weight, out, scratch, stream);
}

#endif // NINFER_TARGET_VOLTA

} // namespace ninfer::ops::detail

#include "ops/linear/projection/nvfp4/nvfp4_dispatch.h"
#include "ops/common/host/operands.h"
#include "ops/linear/projection/nvfp4/nvfp4_shapes.h"
#include "core/platform.h"
#include <array>
#include <stdexcept>

#if NINFER_TARGET_VOLTA
#include "ops/linear/kernels/nvfp4/nvfp4_launch.h"
#endif

namespace ninfer::ops::detail {

#if NINFER_TARGET_VOLTA

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                  LinearPolicy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("nvfp4 linear workspace: invalid token interval");
    if (max_tokens > kNvfp4VoltaQpnMaxTokens && nvfp4_volta_mma_supported(n, k, max_tokens)) {
        return nvfp4_volta_mma_workspace_bytes(n, k, max_tokens);
    }
    return 0;
}

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy,
                    MemoryArena* workspace, cudaStream_t stream) {
    common::validate_nvfp4_weight(weight, "nvfp4 linear");
    const std::int32_t total_t = x.ne[1];
    if (total_t <= 0) throw std::invalid_argument("nvfp4 linear: T must be positive");

    if (workspace != nullptr && total_t > kNvfp4VoltaQpnMaxTokens &&
        nvfp4_volta_mma_supported(weight.n, weight.k, total_t)) {
        const std::size_t need = nvfp4_volta_mma_workspace_bytes(weight.n, weight.k, total_t);
        if (need == 0 || workspace->capacity() - workspace->used() >= need) {
            launch_nvfp4_volta_mma(x, weight, out, *workspace, stream);
            return;
        }
    }

    const bool qpn            = nvfp4_volta_qpn_supported(weight.n, weight.k, kNvfp4VoltaQpnMaxTokens);
    const std::int32_t kChunk = qpn ? kNvfp4VoltaQpnMaxTokens : 32;

    for (std::int32_t token_begin = 0; token_begin < total_t; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, total_t - token_begin);
        auto* input = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(out.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor output_chunk(output, DType::BF16, {weight.n, active});

        if (nvfp4_volta_qpn_supported(weight.n, weight.k, active)) {
            launch_nvfp4_volta_qpn(input_chunk, weight, output_chunk, stream);
        } else if (workspace != nullptr && nvfp4_volta_mma_supported(weight.n, weight.k, active)) {
            launch_nvfp4_volta_mma(input_chunk, weight, output_chunk, *workspace, stream);
        } else {
            MemorySpan empty_storage{nullptr, 0, Device::cpu()};
            MemoryArena dummy(empty_storage);
            launch_nvfp4_volta_mma(input_chunk, weight, output_chunk, dummy, stream);
        }
    }
}

#else

namespace {
const std::array kShapes{
#if NINFER_TARGET_SPARK
    &kNvfp4N12288K4096, &kNvfp4N4096K6144,
#endif
    &kNvfp4N14336K5120, &kNvfp4N16384K5120, &kNvfp4N34816K5120,
    &kNvfp4N5120K6144,  &kNvfp4N5120K17408,
};

const Nvfp4LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("nvfp4 linear: unsupported policy");
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return *shape;
    throw std::invalid_argument("nvfp4 linear: unsupported shape");
}
} // namespace

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("nvfp4 linear workspace: invalid token interval");
    const auto& shape = resolve_shape(n, k, policy);
    return allows_a4(policy) && shape.uses_a4(min_tokens, max_tokens)
               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, k)
               : 0;
}

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    MemoryArena* workspace, cudaStream_t stream) {
    common::validate_nvfp4_weight(weight, "nvfp4 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("nvfp4 linear: T must be positive");
    const auto& shape = resolve_shape(weight.n, weight.k, policy);
    if (!allows_a4(policy) || !shape.uses_a4(x.ne[1], x.ne[1]))
        return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("nvfp4 A4 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    shape.a4(x, weight, out, scratch, stream);
}

#endif // NINFER_TARGET_VOLTA

} // namespace ninfer::ops::detail

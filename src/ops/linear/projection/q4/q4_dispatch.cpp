#include "ops/linear/projection/q4/q4_dispatch.h"
#include "ops/linear/kernels/q4/q4_launch.h"
#include "ops/linear/projection/q4/q4_shapes.h"
#include "core/platform.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {

#if NINFER_TARGET_VOLTA

void q4_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy,
                 cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
    if (t <= 0) throw std::invalid_argument("q4 linear: T must be positive");

    if (q4_uses_volta_qpn(out.ne[0], x.ne[0], t)) {
        launch_q4_volta_qpn(x, w, out, stream);
        return;
    }
    if (q4_volta_mma_supported(out.ne[0], x.ne[0], t)) {
        MemorySpan empty_storage{nullptr, 0, Device::cpu()};
        MemoryArena dummy(empty_storage);
        launch_q4_volta_mma(x, w, out, dummy, stream, 1);
        return;
    }
    throw std::invalid_argument("q4 linear: unsupported shape on Volta");
}

#else

namespace {
struct ShapeEntry {
    std::int32_t n, k;
    Q4Launch (*select)(std::int32_t);
};

constexpr std::array kShapes{
#if NINFER_TARGET_SPARK
    ShapeEntry{77440, 4096, select_q4_n77440_k4096},
#endif
    ShapeEntry{1024, 5120, select_q4_n1024_k5120},
    ShapeEntry{4096, 5120, select_q4_n4096_k5120},
    ShapeEntry{5120, 6144, select_q4_n5120_k6144},
    ShapeEntry{6144, 5120, select_q4_n6144_k5120},
    ShapeEntry{7168, 5120, select_q4_n7168_k5120},
    ShapeEntry{34816, 5120, select_q4_n34816_k5120},
    ShapeEntry{131072, 5120, select_q4_n131072_k5120},
    ShapeEntry{131072, 2048, select_q4_n131072_k2048},
    ShapeEntry{248320, 2560, select_q4_n248320_k2560},
    ShapeEntry{124160, 2560, select_q4_n124160_k2560},
    ShapeEntry{3456, 1152, select_q4_n3456_k1152},
    ShapeEntry{4304, 1152, select_q4_n4304_k1152},
};
} // namespace

Q4Launch select_q4_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q4 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    throw std::invalid_argument("q4 linear: unsupported shape");
}

Q4Launch select_q4_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q4 linear: unsupported policy");
    return select_q4_a16_launch(n, k, t);
}

void q4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q4_launch(weight.n, weight.k, x.ne[1], policy)(x, weight, out, stream);
}

#endif // NINFER_TARGET_VOLTA

} // namespace ninfer::ops::detail

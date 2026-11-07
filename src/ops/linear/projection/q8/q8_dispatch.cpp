#include "ops/linear/projection/q8/q8_dispatch.h"
#include "ops/linear/kernels/q8/q8_launch.h"
#include "core/platform.h"

#include <stdexcept>

namespace ninfer::ops::detail {

#if NINFER_TARGET_VOLTA

void q8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy,
                 cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
    if (t <= 0) throw std::invalid_argument("q8 linear: T must be positive");

    if (w8_volta_qpn_supported(out.ne[0], x.ne[0], t)) {
        launch_w8_volta_qpn(x, w, out, stream);
        return;
    }
    if (w8_volta_mma_supported(out.ne[0], x.ne[0], t)) {
        launch_w8_volta_mma(x, w, out, stream);
        return;
    }
    throw std::invalid_argument("q8 linear: unsupported shape on Volta");
}

Q8Launch select_q8_launch(std::int32_t, std::int32_t, std::int32_t, LinearPolicy) {
    throw std::invalid_argument("q8 linear: use q8_dispatch on Volta");
}

#else

Q8Launch select_q8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q8 linear: unsupported policy");
    return select_q8_a16_launch(n, k, t);
}

void q8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q8_launch(w.n, w.k, x.ne[1], policy)(x, w, out, stream);
}

#endif // NINFER_TARGET_VOLTA

} // namespace ninfer::ops::detail

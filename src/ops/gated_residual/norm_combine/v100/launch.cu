#include "ops/gated_residual/norm_combine/launch.h"
#include <stdexcept>

namespace ninfer::ops::detail::hc {
void launch_norm_combine(Form, Operands, CudaExecutionView) {
    throw std::invalid_argument("gated_residual norm_combine: not supported on Volta (sm_70)");
}
} // namespace ninfer::ops::detail::hc

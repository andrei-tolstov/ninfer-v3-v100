#include "ops/gated_residual/mix/implementation.h"
#include <stdexcept>

namespace ninfer::ops::detail::hc::mix {
Plan select(Form, int, int) {
    throw std::invalid_argument("gated_residual mix select: not supported on Volta (sm_70)");
}
void launch(Form, Operands, Plan, void*, CudaExecutionView) {
    throw std::invalid_argument("gated_residual mix launch: not supported on Volta (sm_70)");
}
} // namespace ninfer::ops::detail::hc::mix

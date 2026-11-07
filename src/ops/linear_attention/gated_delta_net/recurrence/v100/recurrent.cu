#include "ops/linear_attention/gated_delta_net/recurrence/launch.h"
#include <stdexcept>

namespace ninfer::ops::detail::gdn {
void launch_recurrence(Operands, Plan, CudaExecutionView) {
    throw std::invalid_argument("gdn recurrence launch: not supported on Volta (sm_70)");
}
} // namespace ninfer::ops::detail::gdn

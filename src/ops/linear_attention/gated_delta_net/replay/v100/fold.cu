#include "ops/linear_attention/gated_delta_net/replay/launch.h"
#include <stdexcept>

namespace ninfer::ops::detail::gdn {
void launch_fold(Operands, CudaExecutionView) {
    throw std::invalid_argument("gdn fold launch: not supported on Volta (sm_70)");
}
} // namespace ninfer::ops::detail::gdn
